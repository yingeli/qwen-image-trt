import os
import tempfile
import argparse

import onnx
import onnxruntime as ort
import torch
import numpy as np
from diffusers import QwenImageTransformer2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen-Image',
                        help="Qwen-Image model path or hf model id")
    parser.add_argument("--onnx_path", type=str, default='transformer.onnx',
                        help="ONNX path of exported Qwen-Image\'s dit")
    parser.add_argument("--test_resolutions", type=str, default='1664x928',
                        help="Comma-separated list of resolutions to test (e.g., '1664x928,1024x1024,512x512')")
    parser.add_argument("--verify", action='store_true',
                        help="Verify exported ONNX model with multiple resolutions")
    return parser.parse_args()


class TransformerWrapper(torch.nn.Module):
    """Wrapper to handle dynamic img_shapes input for ONNX export"""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_height, img_width, txt_seq_len):
        """
        Args:
            hidden_states: (batch, img_seq_len, in_channels)
            encoder_hidden_states: (batch, txt_seq_len, joint_attention_dim)
            timestep: (batch,)
            img_height: scalar tensor - image height in patches (height // 16)
            img_width: scalar tensor - image width in patches (width // 16)
            txt_seq_len: scalar tensor - text sequence length
        """
        batch_size = hidden_states.shape[0]

        # Convert scalar tensors to Python ints for img_shapes
        h = int(img_height.item())
        w = int(img_width.item())
        txt_len = int(txt_seq_len.item())

        out = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            timestep=timestep,
            img_shapes=[[(batch_size, h, w)]],
            txt_seq_lens=[txt_len],
            guidance=None,
            attention_kwargs=None,
            controlnet_block_samples=None,
            return_dict=False,
        )[0]

        return out


def verify_onnx_model(onnx_path, resolutions, wrapper, dtype, device,
                     in_channels, joint_attention_dim, txt_seq_len):
    """Verify ONNX model with multiple resolutions"""
    print('\n' + '='*60)
    print('Verifying ONNX model with multiple resolutions...')
    print('='*60)

    try:
        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        print(f'ONNX Runtime providers: {session.get_providers()}\n')

        batch_size = 1
        all_passed = True

        for img_width, img_height in resolutions:
            print(f'Testing resolution: {img_width}x{img_height}')

            img_seq_len = img_width // 16 * img_height // 16

            # Create test inputs
            hidden_states = torch.randn(
                (batch_size, img_seq_len, in_channels),
                dtype=dtype,
                device=device,
            )
            timestep = torch.randint(
                0, 1000,
                (batch_size,),
                device=device,
            ).to(dtype=dtype)
            encoder_hidden_states = torch.randn(
                (batch_size, txt_seq_len, joint_attention_dim),
                dtype=dtype,
                device=device,
            )
            img_height_tensor = torch.tensor(img_height // 16, dtype=torch.int64, device=device)
            img_width_tensor = torch.tensor(img_width // 16, dtype=torch.int64, device=device)
            txt_seq_len_tensor = torch.tensor(txt_seq_len, dtype=torch.int64, device=device)

            # PyTorch inference
            with torch.inference_mode():
                torch_output = wrapper(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    img_height_tensor,
                    img_width_tensor,
                    txt_seq_len_tensor,
                )

            # ONNX inference
            onnx_inputs = {
                'hidden_states': hidden_states.cpu().numpy(),
                'encoder_hidden_states': encoder_hidden_states.cpu().numpy(),
                'timestep': timestep.cpu().numpy(),
                'img_height': img_height_tensor.cpu().numpy(),
                'img_width': img_width_tensor.cpu().numpy(),
                'txt_seq_len': txt_seq_len_tensor.cpu().numpy(),
            }

            onnx_output = session.run(None, onnx_inputs)[0]

            # Compare outputs
            torch_output_np = torch_output.cpu().numpy()
            max_diff = np.abs(torch_output_np - onnx_output).max()
            mean_diff = np.abs(torch_output_np - onnx_output).mean()

            print(f'  Output shape: {onnx_output.shape}')
            print(f'  Max difference: {max_diff:.6f}')
            print(f'  Mean difference: {mean_diff:.6f}')

            # Check if differences are acceptable (considering bfloat16 precision)
            if max_diff < 0.1:  # Tolerance for bfloat16
                print(f'  ✓ PASSED\n')
            else:
                print(f'  ✗ FAILED - Difference too large\n')
                all_passed = False

        print('='*60)
        if all_passed:
            print('All resolution tests PASSED!')
        else:
            print('Some resolution tests FAILED!')
        print('='*60)

    except Exception as e:
        print(f'\nVerification failed with error: {e}')
        print('Note: ONNX Runtime may not be installed or configured correctly.')


@torch.inference_mode()
def main(args: argparse.Namespace):
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    transformer: QwenImageTransformer2DModel = QwenImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder='transformer',
        torch_dtype=dtype,
    )

    transformer.eval()
    transformer.to(dtype=dtype, device=device)

    # Parse test resolutions
    resolutions = []
    for res_str in args.test_resolutions.split(','):
        w, h = map(int, res_str.strip().split('x'))
        resolutions.append((w, h))

    # Use first resolution for export
    img_width, img_height = resolutions[0]
    print(f'Exporting with primary resolution: {img_width}x{img_height}')

    batch_size = 1
    img_seq_len = img_width // 16 * img_height // 16
    txt_seq_len = 256
    in_channels = transformer.config.in_channels  # 64
    joint_attention_dim = transformer.config.joint_attention_dim  # 3584

    # Create wrapper model for dynamic resolution support
    wrapper = TransformerWrapper(transformer)
    wrapper.eval()

    # Prepare example inputs for export
    hidden_states = torch.randn(
        (batch_size, img_seq_len, in_channels),
        dtype=dtype,
        device=device,
    )
    timestep = torch.randint(
        0, 1000,
        (batch_size,),
        device=device,
    ).to(dtype=dtype)
    encoder_hidden_states = torch.randn(
        (batch_size, txt_seq_len, joint_attention_dim),
        dtype=dtype,
        device=device,
    )
    img_height_tensor = torch.tensor(img_height // 16, dtype=torch.int64, device=device)
    img_width_tensor = torch.tensor(img_width // 16, dtype=torch.int64, device=device)
    txt_seq_len_tensor = torch.tensor(txt_seq_len, dtype=torch.int64, device=device)

    # Test forward pass
    out_hidden_states = wrapper(
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_height_tensor,
        img_width_tensor,
        txt_seq_len_tensor,
    )

    print(f'Output shape: {out_hidden_states.shape}\n')

    # Export to ONNX with dynamic axes
    with tempfile.TemporaryDirectory() as d:
        temp_path = f'{d}/{os.path.basename(args.onnx_path)}'
        torch.onnx.export(
            wrapper,
            (
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_height_tensor,
                img_width_tensor,
                txt_seq_len_tensor,
            ),
            temp_path,
            opset_version=17,
            input_names=[
                'hidden_states',
                'encoder_hidden_states',
                'timestep',
                'img_height',
                'img_width',
                'txt_seq_len',
            ],
            output_names=['out_hidden_states'],
            dynamic_axes={
                'hidden_states': {1: 'img_seq_len'},
                'encoder_hidden_states': {1: 'txt_seq_len'},
                'out_hidden_states': {1: 'img_seq_len'},
            }
        )
        onnx_model = onnx.load(temp_path)
        onnx.save(
            onnx_model,
            args.onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=args.onnx_path.replace('.onnx', '.onnx.data'),
        )

    print(f'Successfully exported ONNX model to {args.onnx_path}')

    # Verify with multiple resolutions if requested
    if args.verify:
        verify_onnx_model(args.onnx_path, resolutions, wrapper, dtype, device,
                         in_channels, joint_attention_dim, txt_seq_len)


if __name__ == '__main__':
    main(parse_args())