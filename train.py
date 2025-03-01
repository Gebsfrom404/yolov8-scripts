# Imports
import os
import torch
from ultralytics import YOLO
import argparse
import shutil

def get_latest_file():
    try:
        # Получить список всех файлов в папке
        files = [f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))]
        
        if not files:
            return None  # Если файлов нет
        
        # Найти самый последний файл по времени модификации
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join('models', f)))
        
        return latest_file
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def move_and_rename_best_model(source_folder, destination_folder, model_name):
    try:
        # Полный путь к файлу best.pt
        source_file = os.path.join(source_folder, "best.pt")
        
        # Проверяем, существует ли файл
        if not os.path.exists(source_file):
            print(f"Файл {source_file} не найден.")
            return
        
        # Создаем папку назначения, если её нет
        os.makedirs(destination_folder, exist_ok=True)
        
        # Полный путь к новому имени файла
        destination_file = os.path.join(destination_folder, f"{model_name}.pt")
        
        # Перемещаем и переименовываем файл
        shutil.move(source_file, destination_file)
        print(f"Файл {source_file} успешно перемещен в {destination_folder} и переименован в {model_name}.pt.")
    except Exception as e:
        print(f"Ошибка: {e}")
        
def main(folder_name:str,model_type:int=1):
    # Check if CUDA (GPU support) is available
    training_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {training_device}')
    print("Using device:", training_device)
    print(torch.cuda.current_device())
    #exit(0)
    # Load a pretrained model
    # Model Options:
    '''
    yolov8n.pt # Nano Detection
    yolov8s.pt # Small Detection
    yolov8m.pt # Medium Detection
    yolov8l.pt # Large Detection
    yolov8x.pt # Xtra Large Detection

    yolov8n-seg # Nano Segmentation
    yolov8s-seg # Small Segmentation
    yolov8m-seg # Medium Segmentation
    yolov8l-seg # Large Segmentation
    yolov8x-seg # Xtra Large Segmentation
    '''
    # User settings
    output_dir = 'training_output'
    ##folder_name = 'zinfyu'
    match model_type:
        case 1:
            starting_model = 'models/yolov8x.pt'
        case 2:
            starting_model = 'models/yolo11x-obb.pt' #obb model
        case 3:
            starting_model = 'models/yolo11l-seg.pt' #obb model
        #case False if train_latest:
            #starting_model = 'models/' + str(get_latest_file())
        
        case _:
            raise(ValueError('No model to choose from'))

    batch_size = -1 # Batch size for training
    epoch_count = 500 # Number of training epochs

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Absolute path to dataset.yaml
    dataset_path = os.path.abspath('dataset/data.yaml')

    # Run the training
    modelYolo = YOLO(starting_model)
    modelYolo.train(data=dataset_path,
                    epochs=epoch_count,
                    batch=batch_size,
                    device=training_device,
                    project=output_dir,
                    name=folder_name,
                    cache='disk',
                    imgsz=640,
                    deterministic=False,
                    single_cls=True)

    # Evaluate model performance on the validation set
    metrics = modelYolo.val()

    # Optional: Export the model to alternative formats
    # Format Options:
    '''
    Format      	Argument        Model 	                Metadata 	Arguments
    PyTorch 	    - 	            yolov8n.pt 	            yes 	    -
    TorchScript 	torchscript 	yolov8n.torchscript 	yes	        imgsz, optimize
    ONNX 	        onnx 	        yolov8n.onnx 	        yes 	    imgsz, half, dynamic, simplify, opset
    OpenVINO 	    openvino 	    yolov8n_openvino_model/ yes 	    imgsz, half, int8
    TensorRT 	    engine 	        yolov8n.engine 	        yes 	    imgsz, half, dynamic, simplify, workspace
    CoreML 	        coreml 	        yolov8n.mlpackage 	    yes 	    imgsz, half, int8, nms
    TF SavedModel 	saved_model 	yolov8n_saved_model/ 	yes 	    imgsz, keras, int8
    TF GraphDef 	pb 	            yolov8n.pb 	            no 	        imgsz
    TF Lite 	    tflite 	        yolov8n.tflite 	        yes 	    imgsz, half, int8
    TF  Edge TPU 	edgetpu 	    yolov8n_edgetpu.tflite 	yes 	    imgsz
    TF.js 	        tfjs 	        yolov8n_web_model/ 	    yes 	    imgsz, half, int8
    PaddlePaddle 	paddle 	        yolov8n_paddle_model/ 	yes 	    imgsz
    ncnn 	        ncnn 	        yolov8n_ncnn_model/ 	yes 	    imgsz, half
    '''
    # path = model.export(format="onnx") # Export to alternative formats
    move_and_rename_best_model(f'training_output/{folder_name}/weights', 'models', folder_name)

    # Keep the script running (Optional)
    input("Press Enter to exit...")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Training script',
                    description='Script trains simple yolo model')
    parser.add_argument("--output_name", nargs='?', const='default', help="Name of the output model", type=str)
    parser.add_argument("--model_type", nargs='?', const=1, help="Model type, 1-ob, 2-obb, 3-segm", type=int)
    #parser.add_argument('--type',action='store_true')
    #parser.add_argument('--use_latest',action='store_true')
    args = parser.parse_args()
    main(args.output_name,args.model_type)
