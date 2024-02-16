https://www.youtube.com/watch?v=JSnKyGEULdI
![Alt text](image.png)
### ^Inspiration...

# Repo for Market Price Prediction Algorithm Development

## Dev environment setup:
- Python version used: 3.10.5. 
- Recommed using the latest version of python 3 going forward...
- Also recommend virtual environment setup following instructions at the bottom of this readme.
- Standardized Docker dev containers likely coming soon...

## Current State:
- Two simple SVR algorithms and a moving average strategy; the SVRs use the `sklearn` module as the ML backbone
- The repo primarily leverages the python `backtrader`, `yfinance`, `numpy`, and `matplotlib` modules for raw data, formatting, and visualization

## Future:
- Develop an algorithm that approaches the ~70% win rate described in the research below
- Get freaky

## Related Research:

## Setting Up a Python Virtual Environment

To run the project locally, it's recommended to set up a Python virtual environment. This ensures that the project's dependencies are isolated from your system's Python environment.

### Prerequisites

- Python 3.x installed on your system

### Steps

 1. **Clone the Repository**
   ```sh
   git clone https://github.com/Gmunster33/TheLeperColony.git
   cd TheLeperColony
   ```

2. **Create a Virtual Environment**

   Create a new virtual environment in the project directory:

   - On Unix/macOS:
   ```sh
     python3 -m venv venv
   ```

   - On Windows:
    ```powershell
     python -m venv venv
    ```

   This will create a new directory called \`venv\` in your project folder, containing the virtual environment.

3. **Activate the Virtual Environment**

   Before installing dependencies or running the project, you need to activate the virtual environment:

   - On Unix/macOS:

   ```sh
     source venv/bin/activate
   ```
   - On Windows:

     ```powershell
     .\venv\Scripts\activate
     ```

   You'll know the virtual environment is activated when you see \`(venv)\` before your command prompt.

4. **Install Dependencies**

   With the virtual environment activated, install the project dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. **Run the Project**

   Now that the environment is set up and dependencies are installed, you can run the project:

   ```sh
   python your_script.py  # Replace with your script of choice from this repo
   ```

6. **Deactivate the Virtual Environment**

   When you're done working on the project, you can deactivate the virtual environment:

    - On Unix/macOS:

   ```sh
     source venv/bin/deactivate
   ```
   - On Windows:

     ```powershell
     .\venv\Scripts\deactivate
     ```