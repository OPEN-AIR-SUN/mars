# Copyright 2023 Tianyu Liu [tliubk@connect.ust.hk]. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" To generate an instance mask to a nerfstudio compatible dataset."""

from dataclasses import dataclass

from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import (
    BaseConverterToNerfstudioDataset,
)


@dataclass
class GenerateInstanceMaskKittiMot(BaseConverterToNerfstudioDataset):
    """Generate instance mask for kitti mot dataset 
    
    1. load the images, labels and calibration files.
    2. 

    """

