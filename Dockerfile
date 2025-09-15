# Use official Python 3.10 slim base image for small footprint
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt first for optimized caching
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into container
COPY . .

# Expose port (must match Render PORT env or your hosting config)
EXPOSE 10000

# Run Gunicorn server with your Flask app
# Change "ai_alerts_bot_advanced" to your script filename without .py extension
CMD ["gunicorn", "-b", "0.0.0.0:10000", "ai_alerts_bot_advanced:app"]
