# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null
then
    echo "Python 3.12 could not be found. Please install it."
    exit
fi

python3 -m venv venv_data_science_exam

#activate
source ./venv_data_science_exam/bin/activate

# Install requirements

pip3 install -r requirements.txt
