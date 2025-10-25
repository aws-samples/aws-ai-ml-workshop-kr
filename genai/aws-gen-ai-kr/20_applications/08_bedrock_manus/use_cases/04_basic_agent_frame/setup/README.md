# Create UV environment (automatically creates symlinks in root)
cd setup/
./create-uv-env.sh basic-agent-frame 3.12

# Run the project (from root directory)
cd ..
uv run python main.py
```