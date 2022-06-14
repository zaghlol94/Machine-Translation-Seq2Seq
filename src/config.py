config = {
    "src_train": "../data-set/train.de",
    "trg_train":  "../data-set/train.en",
    "src_valid": "../data-set/val.de",
    "trg_valid":  "../data-set/val.en",
    "ENC_EMB_DIM": 300,
    "DEC_EMB_DIM": 300,
    "HID_DIM": 1024,
    "N_LAYERS": 2,
    "ENC_DROPOUT": 0.5,
    "DEC_DROPOUT":  0.5,
    "N_EPOCHS":  20,
    "CLIP": 1,
    "learning_rate": 0.001,
    "test_config": {
        "model_path": "model.pt",
        "src_test": "../data-set/test_2016_flickr.de",
        "trg_test": "../data-set/test_2016_flickr.en",
    }
}
