# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
import torch.nn as nn

import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Load HiDDeN models

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    """
    def __init__(self, num_blocks, num_bits, channels, redundancy=1):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits*redundancy))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits*redundancy, num_bits*redundancy)

        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x)

        x = x.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        x = torch.sum(x, dim=-1) # b k r -> b k

        return x

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs)

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

def get_hidden_decoder(num_bits, redundancy=1, num_blocks=7, channels=64):
    decoder = HiddenDecoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels, redundancy=redundancy)
    return decoder

def get_hidden_decoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    decoder_ckpt = { k.replace('module.', '').replace('decoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'decoder' in k}
    return decoder_ckpt

def get_hidden_encoder(num_bits, num_blocks=4, channels=64):
    encoder = HiddenEncoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels)
    return encoder

def get_hidden_encoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder_ckpt = { k.replace('module.', '').replace('encoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'encoder' in k}
    return encoder_ckpt

### Load LDM models

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



class WMEbeddingNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(WMEbeddingNet, self).__init__()

        self.base_net = nn.Sequential(
                nn.Linear(in_planes, 128),
                nn.LeakyReLU(0.2,inplace=True),
                # nn.Linear(128, 128),
                # nn.LeakyReLU(0.2,inplace=True),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2,inplace=True),
                # nn.Linear(512, out_planes)
                )
        self.head1 = nn.Linear(128, out_planes)

    def forward(self, x):
        x = self.base_net(x)
        x1 = self.head1(x)
        return x1
        

class ConvLoRA(nn.Module):
    def __init__(self, conv_module):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module
        self.embedding_net1 = WMEbeddingNet(32, 256).to(device) # 64->1024、out_planes
        self.embedding_net2 = WMEbeddingNet(32, 256).to(device) # 64->1024、out_planes

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size[0]

        # Actual trainable parameters
        self.lora_A1 = nn.Parameter(self.conv.weight.new_zeros((16, self.in_channels * self.kernel_size)), requires_grad=True)
        self.lora_A2 = nn.Parameter(self.conv.weight.new_zeros((16, self.in_channels * self.kernel_size)), requires_grad=True)
        self.lora_B = nn.Parameter(self.conv.weight.new_zeros((self.out_channels*self.kernel_size, 16)), requires_grad=True)
        # self.lora_B2 = nn.Parameter(self.conv.weight.new_zeros((self.out_channels*self.kernel_size, 16)), requires_grad=True)

        # self.conv.weight.requires_grad = False
        # self.conv.bias.requires_grad = False

        nn.init.normal_(self.lora_A1)
        nn.init.normal_(self.lora_A2)
        # nn.init.normal_(self.lora_B)
        self.if_merged = False
        self.if_robust_ft = False

    def robust_ft(self):
        self.if_robust_ft = True
        self.lora_A1_weight = nn.Parameter(self.conv.weight.new_zeros((16, self.in_channels * self.kernel_size)), requires_grad=True)
        self.lora_A2_weight = nn.Parameter(self.conv.weight.new_zeros((16, self.in_channels * self.kernel_size)), requires_grad=True)
        self.lora_B_weight = nn.Parameter(self.conv.weight.new_zeros((self.out_channels*self.kernel_size, 16)), requires_grad=True)

        self.lora_A1_weight.data = self.lora_A1.data
        self.lora_A2_weight.data = self.lora_A2.data
        self.lora_B_weight.data = self.lora_B.data

    def forward(self, x, wms):
        if not self.if_merged:
            if not self.if_robust_ft:
                batch_size, in_planes, height, width = x.size()
                wm1,wm2 = wms[0],wms[1]
                # wm1,wm2 = torch.randn(1,32).to(device), torch.randn(1,32).to(device)

                wm_coff1 = self.embedding_net1(wm1)
                wm_coff2 = self.embedding_net2(wm2)

                org_output = F.conv2d(x, weight = self.conv.weight, bias = self.conv.bias, stride=1, padding=1, groups=1)


                x = x.reshape(1, -1, height, width)
                # for i in range(batch_size):
                aggregate_weightA = torch.stack(
                    [((self.lora_B @ wm_coff1[i].view((16,16))) @ self.lora_A1).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) for i in range(batch_size)]) 
                aggregate_weightA = aggregate_weightA.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
                outputA = F.conv2d(x, weight=aggregate_weightA, bias=None, stride=1, padding=1, groups=batch_size)
                outputA = outputA.view(batch_size, self.out_channels, outputA.size(-2), outputA.size(-1))

                # for i in range(batch_size):
                #     aggregate_weightB = ((self.lora_B @ wm_coff2[i].view((16,16))) @ self.lora_A2).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) 
                aggregate_weightB = torch.stack(
                    [((self.lora_B @ wm_coff2[i].view((16,16))) @ self.lora_A2).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) for i in range(batch_size)]) 
                aggregate_weightB = aggregate_weightB.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
                outputB = F.conv2d(x, weight=aggregate_weightB,  bias=None, stride=1, padding=1, groups=batch_size)
                outputB = outputB.view(batch_size, self.out_channels, outputB.size(-2), outputB.size(-1))

                return org_output + outputA + outputB
            else:
                batch_size, in_planes, height, width = x.size()
                wm1,wm2 = wms[0],wms[1]

                wm_coff1 = self.embedding_net1(wm1)
                wm_coff2 = self.embedding_net2(wm2)

                org_output = F.conv2d(x, weight = self.conv.weight, bias = self.conv.bias, stride=1, padding=1, groups=1)


                x = x.reshape(1, -1, height, width)
                # for i in range(batch_size):
                aggregate_weightA = torch.stack(
                    [((self.lora_B_weight @ wm_coff1[i].view((16,16))) @ self.lora_A1_weight).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) for i in range(batch_size)]) 
                aggregate_weightA = aggregate_weightA.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
                outputA = F.conv2d(x, weight=aggregate_weightA, bias=None, stride=1, padding=1, groups=batch_size)
                outputA = outputA.view(batch_size, self.out_channels, outputA.size(-2), outputA.size(-1))

                # for i in range(batch_size):
                #     aggregate_weightB = ((self.lora_B @ wm_coff2[i].view((16,16))) @ self.lora_A2).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) 
                aggregate_weightB = torch.stack(
                    [((self.lora_B_weight @ wm_coff2[i].view((16,16))) @ self.lora_A2_weight).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) for i in range(batch_size)]) 
                aggregate_weightB = aggregate_weightB.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
                outputB = F.conv2d(x, weight=aggregate_weightB,  bias=None, stride=1, padding=1, groups=batch_size)
                outputB = outputB.view(batch_size, self.out_channels, outputB.size(-2), outputB.size(-1))

                return org_output + outputA + outputB

        else:

            # cof_bias = torch.ones((batch_size, 1)).cuda()
            # aggregate_bias = self.conv.bias.unsqueeze(dim=0) * cof_bias  

            # Perform a single convolution with the merged weight
            # x = x.view(1, -1, height, width)  # Reshape to batch size * in_channels for grouped conv
            output = F.conv2d(x, weight=self.weight, bias=self.conv.bias, stride=1, padding=1)
            # output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))

            # # Add the bias if present in the original conv layer
            # if self.conv.bias is not None:
            #     bias = self.conv.bias.unsqueeze(0).expand_as(output)
            #     output += bias

        return output
        
    def weights_merged(self, wms):
        self.if_merged = True
        """
        This method merges the original weights with the LoRA weights and performs a single convolution.
        """
        wm1, wm2 = wms[0], wms[1]

        wm_coff1 = self.embedding_net1(wm1)
        wm_coff2 = self.embedding_net2(wm2)

        # Compute the LoRA weight adjustments for both wm_coff1 and wm_coff2
        aggregate_weightA = ((self.lora_B @ wm_coff1[0].view((16, 16))) @ self.lora_A1).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        aggregate_weightB = ((self.lora_B @ wm_coff2[0].view((16, 16))) @ self.lora_A2).view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) 

        # Merge the LoRA weights with the original convolution weight
        merged_weight = self.conv.weight + aggregate_weightA + aggregate_weightB
        self.weight = nn.Parameter(merged_weight.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), requires_grad=True)

