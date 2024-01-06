import base64
import os
from io import BytesIO
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import tqdm
import uvicorn
import shutil
from PIL import Image
import torch.nn as nn
from fastapi import FastAPI
from torch.utils.data import DataLoader
from pyngrok import ngrok
from models.afwm import AFWM_Vitonhd_lrarms, AFWM_Dressescode_lrarms
from models.networks import load_checkpoint, ResUnetGenerator
from options.train_options import TrainOptions
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import nest_asyncio

opt = TrainOptions().parse()


def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        from data.aligned_dataset_vitonhd import AlignedDataset
        dataset = AlignedDataset()
        dataset.byteinitialize(opt, mode='test')
    elif opt.dataset == 'dresscode':
        from data.aligned_dataset_dresscode import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test', stage='warp')
    return dataset


torch.cuda.set_device(opt.local_rank)
device = torch.device(f'cuda:{opt.local_rank}')

# ----------------------------------------------------------------------------------------
if opt.dataset == 'vitonhd':
    warp_model = AFWM_Vitonhd_lrarms(opt, 51)
elif opt.dataset == 'dresscode':
    warp_model = AFWM_Dressescode_lrarms(opt, 51)
warp_model.train()
warp_model.cuda()
load_checkpoint(warp_model, './DressCode/gp-vton_partflow_dresscode_lrarms_resume_1105/PBAFN_warp_epoch_051.pth')
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)
model = warp_model.to(device)
softmax = torch.nn.Softmax(dim=1)

# ----------------------------------------------------------------------------------------
gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.train()
gen_model.cuda()
load_checkpoint(gen_model, './DressCode/gp-vton_gen_dresscode_wskin_wgan_wodenseleg_lrarms_1108/PBAFN_gen_epoch_201.pth')
gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
model_gen = gen_model.to(device)

app = FastAPI()


def convert_png_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string


@app.post("/try-lower")
async def process_image(image: str, cloth: str, type: str, model=model, model_gen=model_gen):
    opt.warproot = ''
    if os.path.exists('sample/' + opt.name):
        shutil.rmtree('sample/' + opt.name)

    os.makedirs('sample/' + opt.name, exist_ok=True)
    with open("./dataset/Dresscode/test_pairs_unpaired_1008.txt", 'w') as file:
        file.truncate(0)
        file.write(image + "_0.png" + " " + cloth + "_1.png" + " " + type)

    train_data = CreateDataset(opt)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False,
                              num_workers=1, pin_memory=True)

    for ii, data in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            pre_clothes_edge = data['edge']
            clothes = data['color']
            clothes = clothes * pre_clothes_edge
            pose = data['pose']

            size = data['color'].size()
            oneHot_size1 = (size[0], 25, size[2], size[3])
            densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
            densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
            densepose = densepose * 2.0 - 1.0
            densepose_fore = data['densepose'] / 24.0

            left_cloth_sleeve_mask = data['flat_clothes_left_mask']
            cloth_torso_mask = data['flat_clothes_middle_mask']
            right_cloth_sleeve_mask = data['flat_clothes_right_mask']

            clothes_left = clothes * left_cloth_sleeve_mask
            clothes_torso = clothes * cloth_torso_mask
            clothes_right = clothes * right_cloth_sleeve_mask

            cloth_parse_for_d = data['flat_clothes_label'].cuda()
            pose = pose.cuda()
            clothes = clothes.cuda()
            clothes_left = clothes_left.cuda()
            clothes_torso = clothes_torso.cuda()
            clothes_right = clothes_right.cuda()
            pre_clothes_edge = pre_clothes_edge.cuda()
            left_cloth_sleeve_mask = left_cloth_sleeve_mask.cuda()
            cloth_torso_mask = cloth_torso_mask.cuda()
            right_cloth_sleeve_mask = right_cloth_sleeve_mask.cuda()
            preserve_mask3 = data['preserve_mask3'].cuda()

            if opt.resolution == 512:
                concat = torch.cat([densepose, pose, preserve_mask3], 1)
                if opt.dataset == 'vitonhd':
                    flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d,
                                     clothes_left, clothes_torso, clothes_right,
                                     left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask,
                                     preserve_mask3)
                elif opt.dataset == 'dresscode':
                    cloth_type = data['flat_clothes_type'].cuda()
                    flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d,
                                     clothes_left, clothes_torso, clothes_right,
                                     left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask,
                                     preserve_mask3, cloth_type)

                last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                    x_full_all, x_edge_full_all, attention_all, seg_list = flow_out
            else:
                densepose_ds = F.interpolate(densepose, scale_factor=0.5, mode='nearest')
                pose_ds = F.interpolate(pose, scale_factor=0.5, mode='nearest')
                preserve_mask3_ds = F.interpolate(preserve_mask3, scale_factor=0.5, mode='nearest')
                concat = torch.cat([densepose_ds, pose_ds, preserve_mask3_ds], 1)

                clothes_ds = F.interpolate(clothes, scale_factor=0.5, mode='bilinear')
                pre_clothes_edge_ds = F.interpolate(pre_clothes_edge, scale_factor=0.5, mode='nearest')
                cloth_parse_for_d_ds = F.interpolate(cloth_parse_for_d, scale_factor=0.5, mode='nearest')
                clothes_left_ds = F.interpolate(clothes_left, scale_factor=0.5, mode='bilinear')
                clothes_torso_ds = F.interpolate(clothes_torso, scale_factor=0.5, mode='bilinear')
                clothes_right_ds = F.interpolate(clothes_right, scale_factor=0.5, mode='bilinear')
                left_cloth_sleeve_mask_ds = F.interpolate(left_cloth_sleeve_mask, scale_factor=0.5, mode='nearest')
                cloth_torso_mask_ds = F.interpolate(cloth_torso_mask, scale_factor=0.5, mode='nearest')
                right_cloth_sleeve_mask_ds = F.interpolate(right_cloth_sleeve_mask, scale_factor=0.5, mode='nearest')

                if opt.dataset == 'vitonhd':
                    flow_out = model(concat, clothes_ds, pre_clothes_edge_ds, cloth_parse_for_d_ds,
                                     clothes_left_ds, clothes_torso_ds, clothes_right_ds,
                                     left_cloth_sleeve_mask_ds, cloth_torso_mask_ds, right_cloth_sleeve_mask_ds,
                                     preserve_mask3_ds)
                elif opt.dataset == 'dresscode':
                    cloth_type = data['flat_clothes_type'].cuda()
                    cloth_type_ds = F.interpolate(cloth_type, scale_factor=0.5, mode='bilinear')
                    flow_out = model(concat, clothes_ds, pre_clothes_edge_ds, cloth_parse_for_d_ds,
                                     clothes_left_ds, clothes_torso_ds, clothes_right_ds,
                                     left_cloth_sleeve_mask_ds, cloth_torso_mask_ds, right_cloth_sleeve_mask_ds,
                                     preserve_mask3_ds, cloth_type_ds)

                last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, x_edge_full_all, attention_all, seg_list = flow_out
                last_flow = F.interpolate(last_flow, scale_factor=2, mode='bilinear')

            bz = pose.size(0)

            left_last_flow = last_flow[0:bz]
            torso_last_flow = last_flow[bz:2 * bz]
            right_last_flow = last_flow[2 * bz:]

            left_warped_full_cloth = F.grid_sample(clothes_left.cuda(), left_last_flow.permute(0, 2, 3, 1),
                                                   mode='bilinear', padding_mode='zeros')
            torso_warped_full_cloth = F.grid_sample(clothes_torso.cuda(), torso_last_flow.permute(0, 2, 3, 1),
                                                    mode='bilinear', padding_mode='zeros')
            right_warped_full_cloth = F.grid_sample(clothes_right.cuda(), right_last_flow.permute(0, 2, 3, 1),
                                                    mode='bilinear', padding_mode='zeros')

            left_warped_cloth_edge = F.grid_sample(left_cloth_sleeve_mask.cuda(), left_last_flow.permute(0, 2, 3, 1),
                                                   mode='nearest', padding_mode='zeros')
            torso_warped_cloth_edge = F.grid_sample(cloth_torso_mask.cuda(), torso_last_flow.permute(0, 2, 3, 1),
                                                    mode='nearest', padding_mode='zeros')
            right_warped_cloth_edge = F.grid_sample(right_cloth_sleeve_mask.cuda(), right_last_flow.permute(0, 2, 3, 1),
                                                    mode='nearest', padding_mode='zeros')

            for bb in range(bz):
                seg_preds = torch.argmax(softmax(seg_list[-1]), dim=1)[:, None, ...].float()
                if opt.resolution == 1024:
                    seg_preds = F.interpolate(seg_preds, scale_factor=2, mode='nearest')

                c_type = data['c_type'][bb]

                if opt.dataset == 'vitonhd':
                    left_mask = (seg_preds[bb] == 1).float()
                    torso_mask = (seg_preds[bb] == 2).float()
                    right_mask = (seg_preds[bb] == 3).float()

                    left_arm_mask = (seg_preds[bb] == 4).float()
                    right_arm_mask = (seg_preds[bb] == 5).float()
                    neck_mask = (seg_preds[bb] == 6).float()

                    warped_cloth_fusion = left_warped_full_cloth[bb] * left_mask + \
                                          torso_warped_full_cloth[bb] * torso_mask + \
                                          right_warped_full_cloth[bb] * right_mask

                    warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 1 + \
                                         torso_warped_cloth_edge[bb] * torso_mask * 2 + \
                                         right_warped_cloth_edge[bb] * right_mask * 3

                    warped_cloth_fusion = warped_cloth_fusion * (1 - preserve_mask3[bb])
                    warped_edge_fusion = warped_edge_fusion * (1 - preserve_mask3[bb])

                    warped_edge_fusion = warped_edge_fusion + \
                                         left_arm_mask * 4 + \
                                         right_arm_mask * 5 + \
                                         neck_mask * 6
                elif opt.dataset == 'dresscode':
                    if c_type == 'upper' or c_type == 'dresses':
                        left_mask = (seg_preds[bb] == 1).float()
                        torso_mask = (seg_preds[bb] == 2).float()
                        right_mask = (seg_preds[bb] == 3).float()
                    else:
                        left_mask = (seg_preds[bb] == 4).float()
                        torso_mask = (seg_preds[bb] == 5).float()
                        right_mask = (seg_preds[bb] == 6).float()

                    left_arms_mask = (seg_preds[bb] == 7).float()
                    right_arms_mask = (seg_preds[bb] == 8).float()
                    neck_mask = (seg_preds[bb] == 9).float()

                    warped_cloth_fusion = left_warped_full_cloth[bb] * left_mask + \
                                          torso_warped_full_cloth[bb] * torso_mask + \
                                          right_warped_full_cloth[bb] * right_mask

                    if c_type == 'upper' or c_type == 'dresses':
                        warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 1 + \
                                             torso_warped_cloth_edge[bb] * torso_mask * 2 + \
                                             right_warped_cloth_edge[bb] * right_mask * 3
                    else:
                        warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 4 + \
                                             torso_warped_cloth_edge[bb] * torso_mask * 5 + \
                                             right_warped_cloth_edge[bb] * right_mask * 6

                    warped_cloth_fusion = warped_cloth_fusion * (1 - preserve_mask3[bb])
                    warped_edge_fusion = warped_edge_fusion * (1 - preserve_mask3[bb])

                    warped_edge_fusion = warped_edge_fusion + \
                                         left_arms_mask * 7 + \
                                         right_arms_mask * 8 + \
                                         neck_mask * 9

                eee = warped_cloth_fusion
                eee_edge = torch.cat([warped_edge_fusion, warped_edge_fusion, warped_edge_fusion], 0)
                eee_edge = eee_edge.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                cv_img = (eee.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                bgr = np.concatenate([bgr, eee_edge], 1)

                cloth_id = data['color_path'][bb].split('/')[-1]
                person_id = data['img_path'][bb].split('/')[-1]
                save_path = 'sample/' + opt.name + '/' + c_type + '___' + person_id + '___' + cloth_id[:-4] + '.png'
                cv2.imwrite(save_path, bgr)

    # ----------------------------------------------------------------------------------------------------------------------
    opt.warproot = './sample/test_lfgp_dresscode_unpaired_1110_v2/'

    train_data = CreateDataset(opt)
    train_loader = DataLoader(train_data, batch_size=12, shuffle=False,
                              num_workers=1, pin_memory=True)

    for data in tqdm.tqdm(train_loader):
        real_image = data['image'].cuda()
        clothes = data['color'].cuda()
        preserve_mask = data['preserve_mask3'].cuda()
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].cuda()
        warped_prod_edge = data['warped_edge'].cuda()
        arms_color = data['arms_color'].cuda()
        arms_neck_label = data['arms_neck_lable'].cuda()
        pose = data['pose'].cuda()

        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)

        gen_outputs = model_gen(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_prod_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        k = p_tryon

        bz = pose.size(0)
        for bb in range(bz):
            combine = k[bb].squeeze()

            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            rgb = (cv_img * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cloth_id = data['color_path'][bb].split('/')[-1]
            person_id = data['img_path'][bb].split('/')[-1]
            c_type = data['c_type'][bb]
            save_path = 'sample/' + opt.name + '/' + c_type + '___' + person_id + '___' + cloth_id[:-4] + '.png'
            cv2.imwrite(save_path, bgr)

            base64_image = convert_png_to_base64(save_path)
            response_data = {
                        "base64_samples": base64_image
                    }

            return response_data


if __name__ == '__main__':
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)