set -e  # Exit on any error

echo "Starting Acne Classification Production App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Load production environment variables
echo "Loading production environment variables..."
export $(cat .env.prod | grep -v '^#' | xargs)

# Create logs directory
mkdir -p web/logs

# Start the application
echo "Starting production server on $HOST:$PORT..."
cd web && python app.py
