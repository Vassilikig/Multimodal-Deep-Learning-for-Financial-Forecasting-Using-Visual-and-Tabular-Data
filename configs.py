class Config:
    """Default configuration parameters for stock return prediction model."""
    
    # Data paths
    train_csv = "./integrated_stock_macro_data/train_dataset.csv"
    val_csv = "./integrated_stock_macro_data/val_dataset.csv"
    test_csv = "./integrated_stock_macro_data/test_dataset.csv"
    data_dir = "./"
    output_dir = "./output"

    
    # Model parameters
    image_embedding_dim = 512
    tabular_embedding_dim = 512
    fusion_hidden_dim = 1024
    dropout = 0.2
    seq_length = 14
    
    # Training parameters
    batch_size = 40  
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 100
    warmup_steps = 500
    early_stopping = True
    patience = 15
    
    # System parameters
    num_workers = 0  
    seed = 42
    resume = ""  # Path to checkpoint if resuming training
