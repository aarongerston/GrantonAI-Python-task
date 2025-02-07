FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-chache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Flask port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP = main.py

COPY .env .env

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app"]