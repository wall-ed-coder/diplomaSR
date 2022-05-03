import os
import torch
import logging
from utils import common_utils as util
from models.second_gan import RRDBNet as net


@torch.no_grad()
def main():
    root_path = '/'.join(os.path.realpath(__file__).split("/")[:-1])
    max_img_size = 210
    borders = 20
    util.logger_info("test_logger")
    logger = logging.getLogger("test_logger")

    testsets = os.path.join(root_path, 'testsets')       # fixed, set path of testsets
    testset_Ls = ['new']  # ['RealSRSet','DPED']

    # model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_names = ['BSRGAN']    # 'BSRGANx2' for scale factor 2

    save_results = True
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join(root_path, 'weights', model_name+'.pth')          # set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        if torch.cuda.is_available():
            logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
            torch.cuda.empty_cache()

        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf, inference=True, max_img_size=max_img_size, borders=borders)  # define network

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for testset_L in testset_Ls:
            L_path = os.path.join(testsets, testset_L)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf))
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))

            for idx, img in enumerate(util.get_image_paths(L_path)):
                try:
                    img_name, ext = os.path.splitext(os.path.basename(img))
                    logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                    img_L = util.imread_uint(img, n_channels=3)
                    img_L = util.uint2tensor4(img_L)
                    img_L = img_L.to(device)

                    img_E = model(img_L)

                    img_E = util.tensor2uint(img_E)
                    if save_results:
                        util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    main()
