## Installation

## Production Setup

To run the application:

### 1. Setup Environment Variables
Create your `.env.prod` file with the required environment variables under /SkinIntelligent_Acne:
```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
PORT=9000
HOST=0.0.0.0
```

### 2. Download the Model
Download the pre-trained model from Hugging Face by running:
```bash
## May take awhile
python install_Model.py
```

### 3. Launch Production Server
Run the startup script to launch the production application:
```bash
./start.sh
```

This script does:
- Create a virtual environment if it doesn't exist
- Install all required dependencies
- Load production environment variables
- Start the server on the configured host and port

## Usage

### Production Web Interface
After running `./start.sh`, the application will be available at http://127.0.0.1/[port] (or your configured host/port).

## Requirements

See `requirements.txt` for complete list.

## Severity Levels

- **Clear**
- **Very Mild**
- **Mild**
- **Moderate**
- **Severe**