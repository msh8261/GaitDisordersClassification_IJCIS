# Motion tracking in diagnosis: Gait disorders classification with a dual-head attentional transformer-LSTM


### windows installation: <br />
- pipenv install <br />
install pytorch with cuda: <br />
- pipenv install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
	

### test keypoints detector:
active virtual environment:
- pipenv shell <br />
test on image file: <br />
- python examples/test_files.py --image_path image_file_name <br />
test on video file: <br />
- python examples/test_files.py --video_path video_file_name <br />

### set config file
- set the parameters in the config file if needed

### prepare dataset
- 1) to make files with features run: python prepare_dataset/1_get_features_frames.py <br />
- 2) to fix sequences in files run: python prepare_dataset/2_fix_sequences.py <br />
- 3) to add augmentation to train files run: python prepare_dataset/3_data_augmentation.py <br />
- 4) to make final datassets run: python prepare_dataset/4_data_for_train.py <br />

### train 
- copy and past the Xtrain.File, Xtest.File, ytrain.File, ytest.File in data folder <br />
- run: python src/train.py <br />

### test trained models
- models should be in models folder <br />
- run: python src/test.py <br />


### models
- Proposed trained model and tpcn model and transformer models are in models folder <br />

### note
- parameters can be set in config file