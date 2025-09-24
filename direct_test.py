

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import csv
import re
import cv2
import glob
import xml.etree.ElementTree as ET
import io
import shutil
import urllib.request
import tarfile
import zipfile
import json
from collections import namedtuple, OrderedDict, defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model configurations
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'download_url': 'http://download.tensorflow.org/models/object_detection/'
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'download_url': 'http://download.tensorflow.org/models/object_detection/'
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'download_url': 'http://download.tensorflow.org/models/object_detection/'
    }
}

class WeaponDetectionSystem:
    def __init__(self, model_type='ssd_mobilenet_v2', project_dir='gun_detection'):
        self.model_type = model_type
        self.project_dir = os.path.abspath(project_dir)
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.train_labels_dir = os.path.join(self.data_dir, 'train_labels')
        self.test_labels_dir = os.path.join(self.data_dir, 'test_labels')
        self.models_dir = os.path.join(self.project_dir, 'models')
        self.research_dir = os.path.join(self.models_dir, 'research')
        self.pretrained_dir = os.path.join(self.research_dir, 'pretrained_model')
        self.training_dir = os.path.join(self.research_dir, 'training')
        self.export_dir = os.path.join(self.research_dir, 'fine_tuned_model')
        
        # Model configuration
        self.model_config = MODELS_CONFIG[model_type]
        self.model_name = self.model_config['model_name']
        self.pipeline_file = self.model_config['pipeline_file']
        
        # Class mapping
        self.class_mapping = {
            'pistol': 1,
            'gun': 1,
            'weapon': 1
        }
    
    def install_requirements(self):
        """Install required packages with error handling"""
        print("Installing required packages...")
        
        packages = [
            'tensorflow>=2.4.0',
            'opencv-python',
            'pandas',
            'numpy',
            'Pillow',
            'lxml',
            'matplotlib',
            'pycocotools',
            'Cython',
            'contextlib2',
            'protobuf<=3.20.3'  # Compatibility fix
        ]
        
        for package in packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✓ Installed {package.split('>=')[0].split('<=')[0]}")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: Failed to install {package}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.project_dir, self.data_dir, self.images_dir,
            self.train_labels_dir, self.test_labels_dir, self.models_dir,
            self.training_dir, self.export_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✓ Created project directories")
    
    def download_dataset(self):
        """Download and extract weapon detection dataset"""
        print("Downloading weapon detection dataset...")
        
        os.chdir(self.project_dir)
        
        try:
            # Download images
            images_url = "https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/WeaponS.zip"
            if not os.path.exists("WeaponS.zip"):
                print("Downloading images...")
                urllib.request.urlretrieve(images_url, "WeaponS.zip")
            
            # Download annotations
            annotations_url = "https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/WeaponS_bbox.zip"
            if not os.path.exists("WeaponS_bbox.zip"):
                print("Downloading annotations...")
                urllib.request.urlretrieve(annotations_url, "WeaponS_bbox.zip")
            
            # Extract files
            print("Extracting files...")
            with zipfile.ZipFile("WeaponS.zip", 'r') as zip_ref:
                zip_ref.extractall("WeaponS_temp")
            
            with zipfile.ZipFile("WeaponS_bbox.zip", 'r') as zip_ref:
                zip_ref.extractall("WeaponS_bbox_temp")
            
            # Move files
            for item in os.listdir("WeaponS_temp"):
                src = os.path.join("WeaponS_temp", item)
                if os.path.isfile(src):
                    shutil.move(src, self.images_dir)
                elif os.path.isdir(src):
                    for file in os.listdir(src):
                        shutil.move(os.path.join(src, file), self.images_dir)
            
            for item in os.listdir("WeaponS_bbox_temp"):
                src = os.path.join("WeaponS_bbox_temp", item)
                if os.path.isfile(src):
                    shutil.move(src, self.train_labels_dir)
                elif os.path.isdir(src):
                    for file in os.listdir(src):
                        shutil.move(os.path.join(src, file), self.train_labels_dir)
            
            # Clean up
            for cleanup in ["WeaponS.zip", "WeaponS_bbox.zip", "WeaponS_temp", "WeaponS_bbox_temp"]:
                if os.path.exists(cleanup):
                    if os.path.isfile(cleanup):
                        os.remove(cleanup)
                    else:
                        shutil.rmtree(cleanup)
            
            print("✓ Dataset downloaded and extracted")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return False
    
    def split_dataset(self, test_ratio=0.2):
        """Split dataset into train/test with proper random seeding"""
        print("Splitting dataset...")
        
        # Get all XML files
        xml_files = glob.glob(os.path.join(self.train_labels_dir, "*.xml"))
        
        if not xml_files:
            print("✗ No XML files found in train_labels directory")
            return False
        
        # Random split
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(xml_files)
        
        split_idx = int(len(xml_files) * (1 - test_ratio))
        test_files = xml_files[split_idx:]
        
        # Move test files
        for xml_file in test_files:
            filename = os.path.basename(xml_file)
            destination = os.path.join(self.test_labels_dir, filename)
            shutil.move(xml_file, destination)
        
        train_count = len(glob.glob(os.path.join(self.train_labels_dir, "*.xml")))
        test_count = len(glob.glob(os.path.join(self.test_labels_dir, "*.xml")))
        
        print(f"✓ Split complete: {train_count} train, {test_count} test")
        return True
    
    def xml_to_csv(self, xml_dir):
        """Convert XML annotations to CSV with improved error handling"""
        xml_list = []
        classes_names = []
        images_extension = 'jpg'
        
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get filename
                filename_elem = root.find('filename')
                if filename_elem is None:
                    continue
                
                filename = filename_elem.text
                if not filename.lower().endswith('.jpg'):
                    filename = f"{filename}.{images_extension}"
                
                # Get image dimensions
                size_elem = root.find('size')
                if size_elem is None:
                    continue
                
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                # Process objects
                for member in root.findall('object'):
                    name_elem = member.find('name')
                    bbox_elem = member.find('bndbox')
                    
                    if name_elem is None or bbox_elem is None:
                        continue
                    
                    class_name = name_elem.text.lower()
                    classes_names.append(class_name)
                    
                    # Get bounding box coordinates
                    xmin = max(0, int(float(bbox_elem.find('xmin').text)))
                    ymin = max(0, int(float(bbox_elem.find('ymin').text)))
                    xmax = min(width, int(float(bbox_elem.find('xmax').text)))
                    ymax = min(height, int(float(bbox_elem.find('ymax').text)))
                    
                    # Validate bounding box
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    
                    value = (filename, width, height, class_name, xmin, ymin, xmax, ymax)
                    xml_list.append(value)
                    
            except Exception as e:
                print(f"⚠ Warning: Error processing {xml_file}: {e}")
                continue
        
        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_names)
        
        # Clean up classes
        classes_names = sorted(list(set(classes_names)))
        
        return xml_df, classes_names
    
    def create_csv_files(self):
        """Create CSV files from XML annotations"""
        print("Creating CSV files...")
        
        # Process training data
        train_df, classes = self.xml_to_csv(self.train_labels_dir)
        train_csv_path = os.path.join(self.data_dir, 'train_labels.csv')
        train_df.to_csv(train_csv_path, index=False)
        print(f"✓ Created train CSV: {len(train_df)} entries")
        
        # Process test data
        test_df, _ = self.xml_to_csv(self.test_labels_dir)
        test_csv_path = os.path.join(self.data_dir, 'test_labels.csv')
        test_df.to_csv(test_csv_path, index=False)
        print(f"✓ Created test CSV: {len(test_df)} entries")
        
        return classes
    
    def create_label_map(self, classes):
        """Create label map file for TensorFlow Object Detection API"""
        print("Creating label map...")
        
        label_map_path = os.path.join(self.data_dir, "label_map.pbtxt")
        
        pbtxt_content = ""
        for i, class_name in enumerate(classes):
            pbtxt_content += f"""item {{
  id: {i + 1}
  name: '{class_name}'
  display_name: '{class_name.title()}'
}}

"""
        
        with open(label_map_path, "w") as f:
            f.write(pbtxt_content.strip())
        
        print(f"✓ Created label map with {len(classes)} classes")
        return label_map_path, len(classes)
    
    def validate_dataset(self):
        """Validate dataset and remove problematic entries"""
        print("Validating dataset...")
        
        problematic_files = []
        
        for csv_name in ['train_labels.csv', 'test_labels.csv']:
            csv_path = os.path.join(self.data_dir, csv_name)
            df = pd.read_csv(csv_path)
            valid_rows = []
            
            for _, row in df.iterrows():
                filename = row['filename']
                img_path = os.path.join(self.images_dir, filename)
                
                # Check if image exists and is readable
                if not os.path.exists(img_path):
                    problematic_files.append(filename)
                    continue
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        problematic_files.append(filename)
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Validate dimensions match
                    if w != row['width'] or h != row['height']:
                        problematic_files.append(filename)
                        continue
                    
                    # Validate bounding box
                    if (row['xmin'] >= w or row['xmax'] >= w or 
                        row['ymin'] >= h or row['ymax'] >= h or
                        row['xmin'] < 0 or row['ymin'] < 0 or
                        row['xmax'] <= row['xmin'] or row['ymax'] <= row['ymin']):
                        problematic_files.append(filename)
                        continue
                    
                    valid_rows.append(row)
                    
                except Exception as e:
                    problematic_files.append(filename)
                    continue
            
            # Save cleaned dataset
            if len(valid_rows) < len(df):
                cleaned_df = pd.DataFrame(valid_rows)
                cleaned_df.to_csv(csv_path, index=False)
                print(f"✓ Cleaned {csv_name}: {len(cleaned_df)}/{len(df)} valid entries")
            else:
                print(f"✓ {csv_name}: All {len(df)} entries valid")
        
        # Remove problematic image files
        for filename in set(problematic_files):
            img_path = os.path.join(self.images_dir, filename)
            if os.path.exists(img_path):
                os.remove(img_path)
        
        if problematic_files:
            print(f"⚠ Removed {len(set(problematic_files))} problematic files")
    
    def setup_tensorflow_models(self):
        """Setup TensorFlow Object Detection API"""
        print("Setting up TensorFlow Object Detection API...")
        
        os.chdir(self.project_dir)
        
        # Clone TensorFlow models if not exists
        if not os.path.exists(self.models_dir):
            try:
                subprocess.run([
                    'git', 'clone', '--quiet',
                    'https://github.com/tensorflow/models.git',
                    self.models_dir
                ], check=True)
                print("✓ Cloned TensorFlow models repository")
            except subprocess.CalledProcessError:
                print("✗ Failed to clone TensorFlow models")
                return False
        
        # Compile protobuf
        os.chdir(self.research_dir)
        try:
            subprocess.run([
                'protoc', 'object_detection/protos/*.proto', '--python_out=.'
            ], shell=True, check=True)
            print("✓ Compiled protobuf files")
        except subprocess.CalledProcessError:
            print("⚠ Protobuf compilation may have failed (continuing anyway)")
        
        # Add to Python path
        if self.research_dir not in sys.path:
            sys.path.insert(0, self.research_dir)
        if os.path.join(self.research_dir, 'slim') not in sys.path:
            sys.path.insert(0, os.path.join(self.research_dir, 'slim'))
        
        return True
    
    def download_pretrained_model(self):
        """Download and setup pretrained model"""
        print("Downloading pretrained model...")
        
        os.chdir(self.research_dir)
        
        model_file = f"{self.model_name}.tar.gz"
        download_url = f"{self.model_config['download_url']}{model_file}"
        
        try:
            # Download if not exists
            if not os.path.exists(model_file):
                urllib.request.urlretrieve(download_url, model_file)
                print(f"✓ Downloaded {model_file}")
            
            # Extract
            with tarfile.open(model_file, 'r:gz') as tar:
                tar.extractall()
            
            # Move to pretrained_model directory
            if os.path.exists(self.pretrained_dir):
                shutil.rmtree(self.pretrained_dir)
            
            shutil.move(self.model_name, self.pretrained_dir)
            os.remove(model_file)
            
            print("✓ Pretrained model setup complete")
            return True
            
        except Exception as e:
            print(f"✗ Error setting up pretrained model: {e}")
            return False
    
    def create_tfrecords(self):
        """Create TFRecord files from CSV data"""
        print("Creating TFRecord files...")
        
        # Import TensorFlow Object Detection utilities
        try:
            from object_detection.utils import dataset_util
        except ImportError:
            print("✗ Cannot import TensorFlow Object Detection API")
            return False
        
        def class_text_to_int(row_label):
            """Convert class text to integer ID"""
            return self.class_mapping.get(row_label.lower(), 1)  # Default to 1
        
        def split_dataframe(df, group):
            """Split dataframe by filename"""
            data = namedtuple('data', ['filename', 'object'])
            gb = df.groupby(group)
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        
        def create_tf_example(group, path):
            """Create TensorFlow Example from grouped data"""
            img_path = os.path.join(path, group.filename)
            
            with tf.io.gfile.GFile(img_path, 'rb') as fid:
                encoded_jpg = fid.read()
            
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size
            
            filename = group.filename.encode('utf8')
            image_format = b'jpg'
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            
            for index, row in group.object.iterrows():
                xmins.append(row['xmin'] / width)
                xmaxs.append(row['xmax'] / width)
                ymins.append(row['ymin'] / height)
                ymaxs.append(row['ymax'] / height)
                classes_text.append(row['class'].encode('utf8'))
                classes.append(class_text_to_int(row['class']))
            
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            
            return tf_example
        
        # Create TFRecords for both train and test
        for dataset_type in ['train_labels', 'test_labels']:
            csv_path = os.path.join(self.data_dir, f'{dataset_type}.csv')
            record_path = os.path.join(self.data_dir, f'{dataset_type}.record')
            
            try:
                df = pd.read_csv(csv_path)
                writer = tf.io.TFRecordWriter(record_path)
                grouped = split_dataframe(df, 'filename')
                
                for group in grouped:
                    try:
                        tf_example = create_tf_example(group, self.images_dir)
                        writer.write(tf_example.SerializeToString())
                    except Exception as e:
                        print(f"⚠ Warning: Skipping {group.filename}: {e}")
                        continue
                
                writer.close()
                print(f"✓ Created {dataset_type}.record")
                
            except Exception as e:
                print(f"✗ Error creating {dataset_type}.record: {e}")
                return False
        
        return True
    
    def create_training_config(self, num_classes, num_steps=50000):
        """Create training configuration file"""
        print("Creating training configuration...")
        
        config_path = os.path.join(self.research_dir, 'pipeline.config')
        
        train_record_path = os.path.join(self.data_dir, 'train_labels.record')
        test_record_path = os.path.join(self.data_dir, 'test_labels.record')
        label_map_path = os.path.join(self.data_dir, 'label_map.pbtxt')
        checkpoint_path = os.path.join(self.pretrained_dir, 'model.ckpt')
        
        # Get test dataset size
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))
        num_examples = len(test_df['filename'].unique())
        
        config_content = f"""model {{
  ssd {{
    num_classes: {num_classes}
    box_coder {{
      faster_rcnn_box_coder {{
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }}
    }}
    matcher {{
      argmax_matcher {{
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }}
    }}
    similarity_calculator {{
      iou_similarity {{
      }}
    }}
    anchor_generator {{
      ssd_anchor_generator {{
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
      }}
    }}
    image_resizer {{
      fixed_shape_resizer {{
        height: 300
        width: 300
      }}
    }}
    box_predictor {{
      convolutional_box_predictor {{
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: true
        dropout_keep_probability: 0.8
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
        conv_hyperparams {{
          activation: RELU_6,
          regularizer {{
            l2_regularizer {{
              weight: 0.001
            }}
          }}
          initializer {{
            truncated_normal_initializer {{
              stddev: 0.03
              mean: 0.0
            }}
          }}
          batch_norm {{
            train: true,
            scale: true,
            center: true,
            decay: 0.9997,
            epsilon: 0.001,
          }}
        }}
      }}
    }}
    feature_extractor {{
      type: 'ssd_mobilenet_v2'
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {{
        activation: RELU_6,
        regularizer {{
          l2_regularizer {{
            weight: 0.001
          }}
        }}
        initializer {{
          truncated_normal_initializer {{
            stddev: 0.03
            mean: 0.0
          }}
        }}
        batch_norm {{
          train: true,
          scale: true,
          center: true,
          decay: 0.9997,
          epsilon: 0.001,
        }}
      }}
    }}
    loss {{
      classification_loss {{
        weighted_sigmoid {{
        }}
      }}
      localization_loss {{
        weighted_smooth_l1 {{
        }}
      }}
      hard_example_miner {{
        num_hard_examples: 3000
        iou_threshold: 0.95
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 3
      }}
      classification_weight: 1.0
      localization_weight: 1.0
    }}
    normalize_loss_by_num_matches: true
    post_processing {{
      batch_non_max_suppression {{
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 16
        max_total_detections: 16
      }}
      score_converter: SIGMOID
    }}
  }}
}}

train_config: {{
  batch_size: 16
  optimizer {{
    rms_prop_optimizer: {{
      learning_rate: {{
        exponential_decay_learning_rate {{
          initial_learning_rate: 0.003
          decay_steps: 800720
          decay_factor: 0.95
        }}
      }}
      momentum_optimizer_value: 0.9
      decay: 0.9
      epsilon: 1.0
    }}
  }}
  fine_tune_checkpoint: "{checkpoint_path}"
  fine_tune_checkpoint_type: "detection"
  num_steps: {num_steps}
  data_augmentation_options {{
    random_horizontal_flip {{
    }}
  }}
  data_augmentation_options {{
    random_adjust_contrast {{
    }}
  }}
  data_augmentation_options {{
    ssd_random_crop {{
    }}
  }}
}}

train_input_reader: {{
  tf_record_input_reader {{
    input_path: "{train_record_path}"
  }}
  label_map_path: "{label_map_path}"
}}

eval_config: {{
  num_examples: {num_examples}
  num_visualizations: 20
}}

eval_input_reader: {{
  tf_record_input_reader {{
    input_path: "{test_record_path}"
  }}
  label_map_path: "{label_map_path}"
  shuffle: false
  num_readers: 1
}}
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Created training config: {config_path}")
        return config_path
    
    def train_model(self, config_path, num_steps=50000):
        """Start model training"""
        print("Starting model training...")
        
        # Clean training directory
        if os.path.exists(self.training_dir):
            shutil.rmtree(self.training_dir)
        os.makedirs(self.training_dir, exist_ok=True)
        
        os.chdir(self.research_dir)
        
        # Training command
        train_cmd = [
            'python3', 'object_detection/model_main.py',
            f'--pipeline_config_path={config_path}',
            f'--model_dir={self.training_dir}',
            '--alsologtostderr'
        ]
        
        print("Training command:", ' '.join(train_cmd))
        print("Training started. This will take a while...")
        print("You can monitor progress in TensorBoard")
        
        try:
            # Start training (this will run for a long time)
            subprocess.run(train_cmd, check=True)
            print("✓ Training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed: {e}")
            return False
    
    def export_model(self):
        """Export trained model for inference"""
        print("Exporting model for inference...")
        
        os.chdir(self.research_dir)
        
        # Find latest checkpoint
        checkpoint_files = glob.glob(os.path.join(self.training_dir, "model.ckpt-*.meta"))
        
        if not checkpoint_files:
            print("✗ No checkpoint files found")
            return False
        
        # Extract step numbers and find the latest
        steps = []
        for file in checkpoint_files:
            match = re.search(r'model\.ckpt-(\d+)\.meta', file)
            if match:
                steps.append(int(match.group(1)))
        
        if not steps:
            print("✗ Could not parse checkpoint step numbers")
            return False
        
        latest_step = max(steps)
        checkpoint_path = os.path.join(self.training_dir, f"model.ckpt-{latest_step}")
        
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Export inference graph
        config_path = os.path.join(self.research_dir, 'pipeline.config')
        
        export_cmd = [
            'python3', 'object_detection/export_inference_graph.py',
            '--input_type=image_tensor',
            f'--pipeline_config_path={config_path}',
            f'--trained_checkpoint_prefix={checkpoint_path}',
            f'--output_directory={self.export_dir}'
        ]
        
        try:
            subprocess.run(export_cmd, check=True)
            print(f"✓ Model exported to: {self.export_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Export failed: {e}")
            return False
    
    def setup_inference(self):
        """Setup inference components"""
        print("Setting up inference...")
        
        # Paths for inference
        frozen_graph_path = os.path.join(self.export_dir, 'frozen_inference_graph.pb')
        label_map_path = os.path.join(self.data_dir, 'label_map.pbtxt')
        
        if not os.path.exists(frozen_graph_path):
            print("✗ Frozen inference graph not found. Run export_model() first.")
            return None
        
        if not os.path.exists(label_map_path):
            print("✗ Label map not found.")
            return None
        
        try:
            # Import required modules
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as vis_util
            
            # Load frozen graph
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(frozen_graph_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            
            # Load label map
            label_map = label_map_util.load_labelmap(label_map_path)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=10, use_display_name=True
            )
            category_index = label_map_util.create_category_index(categories)
            
            print("✓ Inference setup complete")
            
            return {
                'graph': detection_graph,
                'category_index': category_index,
                'frozen_graph_path': frozen_graph_path,
                'label_map_path': label_map_path
            }
            
        except ImportError:
            print("✗ Cannot import TensorFlow Object Detection API utilities")
            return None
        except Exception as e:
            print(f"✗ Error setting up inference: {e}")
            return None
    
    def detect_weapons(self, image_path, inference_components, confidence_threshold=0.5):
        """Detect weapons in a single image"""
        if inference_components is None:
            print("✗ Inference components not initialized")
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 4:
                image_np = image_np[:, :, :3]
            
            # Expand dimensions for model input
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            # Run detection
            graph = inference_components['graph']
            category_index = inference_components['category_index']
            
            with graph.as_default():
                with tf.compat.v1.Session(graph=graph) as sess:
                    # Get input and output tensors
                    input_tensor = graph.get_tensor_by_name('image_tensor:0')
                    output_tensors = [
                        graph.get_tensor_by_name('detection_boxes:0'),
                        graph.get_tensor_by_name('detection_scores:0'),
                        graph.get_tensor_by_name('detection_classes:0'),
                        graph.get_tensor_by_name('num_detections:0')
                    ]
                    
                    # Run inference
                    (boxes, scores, classes, num_detections) = sess.run(
                        output_tensors,
                        feed_dict={input_tensor: image_np_expanded}
                    )
            
            # Process results
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            
            # Filter by confidence
            valid_detections = scores > confidence_threshold
            filtered_boxes = boxes[valid_detections]
            filtered_scores = scores[valid_detections]
            filtered_classes = classes[valid_detections]
            
            return {
                'image': image_np,
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'classes': filtered_classes,
                'category_index': category_index
            }
            
        except Exception as e:
            print(f"✗ Error during detection: {e}")
            return None
    
    def visualize_detections(self, detection_results, save_path=None, show_image=True):
        """Visualize detection results"""
        if detection_results is None:
            return
        
        try:
            from object_detection.utils import visualization_utils as vis_util
            
            image_copy = detection_results['image'].copy()
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_copy,
                detection_results['boxes'],
                detection_results['classes'],
                detection_results['scores'],
                detection_results['category_index'],
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.1
            )
            
            if save_path:
                plt.figure(figsize=(12, 8))
                plt.imshow(image_copy)
                plt.axis('off')
                plt.title('Weapon Detection Results')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"✓ Results saved to: {save_path}")
            
            if show_image:
                plt.figure(figsize=(12, 8))
                plt.imshow(image_copy)
                plt.axis('off')
                plt.title('Weapon Detection Results')
                plt.show()
            
            # Print detection summary
            num_detections = len(detection_results['scores'])
            if num_detections > 0:
                print(f"Detected {num_detections} weapon(s):")
                for i, (score, class_id) in enumerate(zip(detection_results['scores'], detection_results['classes'])):
                    class_name = detection_results['category_index'][class_id]['name']
                    print(f"  {i+1}. {class_name}: {score:.2%} confidence")
            else:
                print("No weapons detected.")
            
        except ImportError:
            print("✗ Cannot import visualization utilities")
        except Exception as e:
            print(f"✗ Error visualizing results: {e}")
    
    def test_inference(self, inference_components, num_test_images=5):
        """Test inference on sample images"""
        print("Testing inference...")
        
        if inference_components is None:
            print("✗ Inference components not available")
            return
        
        # Get test images
        test_images = glob.glob(os.path.join(self.images_dir, "*.jpg"))[:num_test_images]
        
        if not test_images:
            print("✗ No test images found")
            return
        
        for i, img_path in enumerate(test_images):
            print(f"Testing image {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
            
            # Run detection
            results = self.detect_weapons(img_path, inference_components)
            
            if results:
                # Save visualization
                output_path = os.path.join(self.project_dir, f"detection_result_{i+1}.jpg")
                self.visualize_detections(results, save_path=output_path, show_image=False)
            
        print("✓ Inference testing complete")
    
    def run_complete_pipeline(self, num_steps=10000):
        """Run the complete weapon detection pipeline"""
        print("="*60)
        print("COMPLETE WEAPON DETECTION SYSTEM")
        print("="*60)
        
        try:
            # Phase 1: Setup and Data Preparation
            print("\n[PHASE 1: SETUP AND DATA PREPARATION]")
            self.install_requirements()
            self.setup_directories()
            
            if not self.download_dataset():
                return False
            
            if not self.split_dataset():
                return False
            
            classes = self.create_csv_files()
            label_map_path, num_classes = self.create_label_map(classes)
            self.validate_dataset()
            
            print(f"✓ Dataset ready: {num_classes} classes detected")
            
            # Phase 2: Model Setup
            print("\n[PHASE 2: MODEL SETUP]")
            if not self.setup_tensorflow_models():
                return False
            
            if not self.download_pretrained_model():
                return False
            
            if not self.create_tfrecords():
                return False
            
            config_path = self.create_training_config(num_classes, num_steps)
            print("✓ Model setup complete")
            
            # Phase 3: Training (Optional - can be run separately)
            print("\n[PHASE 3: TRAINING]")
            print("Training can take several hours. To start training, run:")
            print(f"  system.train_model('{config_path}', {num_steps})")
            
            # Phase 4: Inference Setup
            print("\n[PHASE 4: INFERENCE SETUP]")
            print("After training completes, you can:")
            print("1. Export the model: system.export_model()")
            print("2. Setup inference: inference_components = system.setup_inference()")
            print("3. Test detection: system.test_inference(inference_components)")
            
            print("\n" + "="*60)
            print("PIPELINE SETUP COMPLETE!")
            print("="*60)
            print(f"Classes: {classes}")
            print(f"Training config: {config_path}")
            print(f"Project directory: {self.project_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class WeaponDetectionDemo:
    """Demo class for easy usage"""
    
    def __init__(self, project_dir='gun_detection'):
        self.system = WeaponDetectionSystem(project_dir=project_dir)
        self.inference_components = None
    
    def quick_setup(self):
        """Quick setup for demonstration"""
        print("Running quick setup...")
        return self.system.run_complete_pipeline(num_steps=5000)  # Reduced steps for demo
    
    def train(self, num_steps=10000):
        """Train the model"""
        config_path = os.path.join(self.system.research_dir, 'pipeline.config')
        return self.system.train_model(config_path, num_steps)
    
    def setup_detection(self):
        """Setup for detection"""
        self.system.export_model()
        self.inference_components = self.system.setup_inference()
        return self.inference_components is not None
    
    def detect_in_image(self, image_path, confidence=0.5):
        """Detect weapons in an image"""
        if self.inference_components is None:
            print("Run setup_detection() first")
            return None
        
        return self.system.detect_weapons(image_path, self.inference_components, confidence)
    
    def visualize(self, detection_results, save_path=None):
        """Visualize detection results"""
        self.system.visualize_detections(detection_results, save_path)


# Main execution
def main():
    """Main function for easy execution"""
    print("Weapon Detection System")
    print("=" * 50)
    
    # Create system instance
    demo = WeaponDetectionDemo()
    
    # Run setup
    if demo.quick_setup():
        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print("1. Train: demo.train()")
        print("2. Setup detection: demo.setup_detection()")
        print("3. Detect: demo.detect_in_image('path/to/image.jpg')")
    else:
        print("Setup failed!")

if __name__ == "__main__":
    main()