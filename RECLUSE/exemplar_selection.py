# This code selects the exemplars from the dataset after each incremental task

import os
import shutil
import re
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image