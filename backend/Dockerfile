FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--reload", "--workers", "4"]
