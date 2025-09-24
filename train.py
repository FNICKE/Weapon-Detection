
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
from collections import namedtuple, OrderedDict
from PIL import Image

# Install required packages
def install_requirements():
    """Install required packages"""
    packages = [
        'tensorflow>=2.0.0',
        'opencv-python',
        'pandas',
        'numpy',
        'Pillow',
        'lxml',
        'matplotlib',
        'pycocotools',
        'Cython',
        'contextlib2'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

# Configuration for different models
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
    }
}

class WeaponDetectionTrainer:
    def __init__(self, model_type='ssd_mobilenet_v2', project_dir='gun_detection'):
        self.model_type = model_type
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, 'data')
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.train_labels_dir = os.path.join(self.data_dir, 'train_labels')
        self.test_labels_dir = os.path.join(self.data_dir, 'test_labels')
        
        # Model configuration
        self.model_config = MODELS_CONFIG[model_type]
        self.model_name = self.model_config['model_name']
        self.pipeline_file = self.model_config['pipeline_file']
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.project_dir,
            self.data_dir,
            self.images_dir,
            self.train_labels_dir,
            self.test_labels_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
    
    def download_dataset(self):
        """Download and extract the weapon detection dataset"""
        print("Downloading weapon detection dataset...")
        
        # URLs for the dataset
        images_url = "https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/WeaponS.zip"
        annotations_url = "https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/WeaponS_bbox.zip"
        
        try:
            # Download images
            print("Downloading images...")
            urllib.request.urlretrieve(images_url, "WeaponS.zip")
            
            # Download annotations
            print("Downloading annotations...")
            urllib.request.urlretrieve(annotations_url, "WeaponS_bbox.zip")
            
            # Extract images
            print("Extracting images...")
            shutil.unpack_archive("WeaponS.zip", "WeaponS")
            
            # Extract annotations
            print("Extracting annotations...")
            shutil.unpack_archive("WeaponS_bbox.zip", "WeaponS_bbox")
            
            # Move files to appropriate directories
            for file in glob.glob("WeaponS/*"):
                shutil.move(file, self.images_dir)
            
            for file in glob.glob("WeaponS_bbox/*"):
                shutil.move(file, self.train_labels_dir)
            
            # Clean up
            os.remove("WeaponS.zip")
            os.remove("WeaponS_bbox.zip")
            shutil.rmtree("WeaponS", ignore_errors=True)
            shutil.rmtree("WeaponS_bbox", ignore_errors=True)
            
            print("✓ Dataset downloaded and extracted successfully")
            
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return False
        
        return True
    
    def split_dataset(self, test_ratio=0.2):
        """Split dataset into training and testing sets"""
        print("Splitting dataset...")
        
        # Get all annotation files
        annotation_files = glob.glob(os.path.join(self.train_labels_dir, "*.xml"))
        
        # Calculate split
        total_files = len(annotation_files)
        test_count = int(total_files * test_ratio)
        
        # Randomly select test files
        np.random.shuffle(annotation_files)
        test_files = annotation_files[:test_count]
        
        # Move test files
        for file_path in test_files:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.test_labels_dir, filename)
            shutil.move(file_path, destination)
        
        train_count = len(glob.glob(os.path.join(self.train_labels_dir, "*.xml")))
        test_count = len(glob.glob(os.path.join(self.test_labels_dir, "*.xml")))
        
        print(f"✓ Training files: {train_count}")
        print(f"✓ Testing files: {test_count}")
    
    def xml_to_csv(self, path, images_extension='jpg'):
        """Convert XML annotations to CSV format"""
        classes_names = []
        xml_list = []
        
        for xml_file in glob.glob(os.path.join(path, '*.xml')):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                filename_elem = root.find('filename')
                size_elem = root.find('size')
                
                if filename_elem is None or size_elem is None:
                    print(f"Warning: Skipping malformed XML file: {xml_file}")
                    continue
                
                filename = filename_elem.text
                if not filename.endswith(f'.{images_extension}'):
                    filename = f"{filename}.{images_extension}"
                
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                for member in root.findall('object'):
                    name_elem = member.find('name')
                    bbox_elem = member.find('bndbox')
                    
                    if name_elem is None or bbox_elem is None:
                        continue
                    
                    class_name = name_elem.text
                    classes_names.append(class_name)
                    
                    xmin = int(float(bbox_elem.find('xmin').text))
                    ymin = int(float(bbox_elem.find('ymin').text))
                    xmax = int(float(bbox_elem.find('xmax').text))
                    ymax = int(float(bbox_elem.find('ymax').text))
                    
                    value = (filename, width, height, class_name, xmin, ymin, xmax, ymax)
                    xml_list.append(value)
                    
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
        
        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_names)
        classes_names = list(set(classes_names))
        classes_names.sort()
        
        return xml_df, classes_names
    
    def create_csv_files(self):
        """Create CSV files from XML annotations"""
        print("Creating CSV files...")
        
        # Process training labels
        train_df, classes = self.xml_to_csv(self.train_labels_dir)
        train_csv_path = os.path.join(self.data_dir, 'train_labels.csv')
        train_df.to_csv(train_csv_path, index=False)
        print(f"✓ Created training CSV: {train_csv_path}")
        
        # Process testing labels
        test_df, _ = self.xml_to_csv(self.test_labels_dir)
        test_csv_path = os.path.join(self.data_dir, 'test_labels.csv')
        test_df.to_csv(test_csv_path, index=False)
        print(f"✓ Created testing CSV: {test_csv_path}")
        
        return classes
    
    def create_label_map(self, classes):
        """Create label map file"""
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
        
        print(f"✓ Created label map: {label_map_path}")
        return label_map_path
    
    def validate_data(self):
        """Validate dataset integrity"""
        print("Validating dataset...")
        
        for csv_file in ['train_labels.csv', 'test_labels.csv']:
            csv_path = os.path.join(self.data_dir, csv_file)
            error_count = 0
            total_count = 0
            
            try:
                df = pd.read_csv(csv_path)
                
                for _, row in df.iterrows():
                    total_count += 1
                    filename = row['filename']
                    image_path = os.path.join(self.images_dir, filename)
                    
                    # Check if image exists
                    if not os.path.exists(image_path):
                        print(f"Missing image: {filename}")
                        error_count += 1
                        continue
                    
                    # Check image dimensions
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"Cannot read image: {filename}")
                            error_count += 1
                            continue
                        
                        h, w = img.shape[:2]
                        
                        # Validate bounding box coordinates
                        if (row['xmin'] >= w or row['xmax'] >= w or 
                            row['ymin'] >= h or row['ymax'] >= h or
                            row['xmin'] < 0 or row['ymin'] < 0):
                            print(f"Invalid bounding box in {filename}")
                            error_count += 1
                            
                    except Exception as e:
                        print(f"Error validating {filename}: {e}")
                        error_count += 1
                
                print(f"✓ {csv_file}: {total_count - error_count}/{total_count} valid entries")
                
            except Exception as e:
                print(f"✗ Error validating {csv_file}: {e}")
    
    def setup_tensorflow_models(self):
        """Setup TensorFlow Object Detection API"""
        print("Setting up TensorFlow Object Detection API...")
        
        models_dir = os.path.join(self.project_dir, 'models')
        
        if not os.path.exists(models_dir):
            try:
                # Clone TensorFlow models repository
                subprocess.run([
                    'git', 'clone', 
                    'https://github.com/tensorflow/models.git',
                    models_dir
                ], check=True)
                print("✓ Cloned TensorFlow models repository")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to clone repository: {e}")
                return False
        
        # Compile protobuf files
        research_dir = os.path.join(models_dir, 'research')
        try:
            subprocess.run([
                'protoc', 'object_detection/protos/*.proto', 
                '--python_out=.'
            ], cwd=research_dir, shell=True, check=True)
            print("✓ Compiled protobuf files")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to compile protobuf: {e}")
        
        # Add to Python path
        sys.path.append(research_dir)
        sys.path.append(os.path.join(research_dir, 'slim'))
        
        return True
    
    def create_config_file(self, num_classes, num_steps=50000):
        """Create training configuration file"""
        print("Creating configuration file...")
        
        config_content = f'''model {{
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
          activation: RELU_6
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
            train: true
            scale: true
            center: true
            decay: 0.9997
            epsilon: 0.001
          }}
        }}
      }}
    }}
    feature_extractor {{
      type: 'ssd_mobilenet_v2'
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {{
        activation: RELU_6
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
          train: true
          scale: true
          center: true
          decay: 0.9997
          epsilon: 0.001
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
  fine_tune_checkpoint: "{os.path.join(self.project_dir, 'pretrained_model', 'model.ckpt')}"
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
    input_path: "{os.path.join(self.data_dir, 'train_labels.record')}"
  }}
  label_map_path: "{os.path.join(self.data_dir, 'label_map.pbtxt')}"
}}

eval_config: {{
  num_examples: 500
  num_visualizations: 20
}}

eval_input_reader: {{
  tf_record_input_reader {{
    input_path: "{os.path.join(self.data_dir, 'test_labels.record')}"
  }}
  label_map_path: "{os.path.join(self.data_dir, 'label_map.pbtxt')}"
  shuffle: false
  num_readers: 1
}}
'''
        
        config_path = os.path.join(self.project_dir, 'pipeline.config')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Created config file: {config_path}")
        return config_path
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("="*50)
        print("WEAPON DETECTION TRAINING PIPELINE")
        print("="*50)
        
        try:
            # Step 1: Setup
            self.setup_directories()
            
            # Step 2: Download dataset
            if not self.download_dataset():
                return False
            
            # Step 3: Split dataset
            self.split_dataset()
            
            # Step 4: Create CSV files
            classes = self.create_csv_files()
            
            # Step 5: Create label map
            self.create_label_map(classes)
            
            # Step 6: Validate data
            self.validate_data()
            
            # Step 7: Setup TensorFlow models
            if not self.setup_tensorflow_models():
                return False
            
            # Step 8: Create config file
            config_path = self.create_config_file(len(classes))
            
            print("="*50)
            print("✓ PIPELINE SETUP COMPLETE!")
            print("="*50)
            print(f"Classes detected: {classes}")
            print(f"Config file: {config_path}")
            print("\nNext steps:")
            print("1. Download pretrained model")
            print("2. Create TFRecord files")
            print("3. Start training")
            
            return True
            
        except Exception as e:
            print(f"✗ Pipeline failed: {e}")
            return False

def main():
    """Main execution function"""
    # Install requirements
    print("Installing requirements...")
    install_requirements()
    
    # Initialize trainer
    trainer = WeaponDetectionTrainer()
    
    # Run pipeline
    success = trainer.run_full_pipeline()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now proceed with model training.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()