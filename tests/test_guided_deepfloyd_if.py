import argparse

import torch
import matplotlib.pyplot as plt

import firefly.api as api

import test_utils as tutils


class TestGuidedDeepFloydIF(tutils.TungstenDataTester):
    
    @tutils.TungstenDataTester.wrap_recon_tester(name="guided_deepfloyd_if")
    def test_guided_deepfloyd_if(self):
        self.atol = 1e-1
        self.rtol = 0
        self.trigger_on_mean_abs_diff = True
        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        inds_to_keep = tutils.get_indices_to_keep_for_downsampling(
            n_total=data.shape[0],
            n_keep=20,
            seed=123
        )
        data = data[inds_to_keep]
        positions_px = positions_px[inds_to_keep]
        
        options = api.GuidedDiffusionOptions()

        options.data_options.data = data

        options.object_options.initial_guess = torch.ones([1, 1024, 1024], dtype=torch.complex64)
        options.object_options.pixel_size_m = pixel_size_m

        options.probe_options.initial_guess = probe

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]

        options.reconstructor_options.num_inference_steps = 100
        options.reconstructor_options.text_guidance_scale = 0
        options.reconstructor_options.physical_guidance_scale = 2e3
        options.reconstructor_options.physical_guidance_method = api.enums.PhysicalGuidanceMethods.SCORE
        options.reconstructor_options.time_travel_plan.stride = torch.inf
        options.reconstructor_options.prompt = ""
        options.reconstructor_options.model_path = "deepfloyd"

        task = api.GuidedDiffusionPtychographyTask(options)
        task.run()

        recon = task.reconstructor.parameter_group.object.data.detach().cpu().numpy()[0, 480:544, 480:544]
        return recon
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestGuidedDeepFloydIF()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_guided_deepfloyd_if()
