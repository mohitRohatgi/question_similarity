class Config:
    label_seq_length = 10
    text_seq_length = 60
    batch_size = 32
    n_epoch = 30
    train_valid_split = 0.8
    evaluate_every = 100
    learning_rate = 1e-4
    gradient_clip_norm = 100
    embeddings_dim = 300
    vocab_size = 96
    num_hidden = 256
    lstm_layers = 3
    dropout_keep = 1.0
