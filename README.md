# CadQuery Web IDE

A browser-based IDE for [CadQuery](https://cadquery.readthedocs.io/) - the Python CAD modeling library.

Write CadQuery Python code in your browser, click Build, and see your 3D model instantly.

## Features

- **Monaco Editor** - VS Code's editor with Python syntax highlighting and autocomplete
- **Real-time 3D Preview** - Interactive Three.js viewer with orbit controls
- **STL Export** - Download models for 3D printing
- **12 Example Projects** - Learn CadQuery with included examples
- **File Open/Save** - Load and save your Python scripts
- **Docker Ready** - Production deployment with security hardening

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/tabibazar/cadquery-web-ide.git
cd cadquery-web-ide

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Open http://localhost:8000 in your browser.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t cadquery-ide .
docker run -p 8000:8000 cadquery-ide
```

## Usage

1. Write CadQuery code in the editor
2. Assign your final object to `result`:
   ```python
   result = cq.Workplane("XY").box(10, 20, 30)
   ```
3. Click **Build** (or press Ctrl+Enter)
4. View the 3D model in the viewer
5. Click **STL** to download for 3D printing

> **Note:** `cq` and `math` are pre-imported. No import statements needed.

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` or `production` |
| `APP_PORT` | `8000` | Server port |
| `APP_WORKERS` | `2` | Gunicorn workers (production) |
| `EXECUTION_TIMEOUT` | `30` | Max code execution time (seconds) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

## Project Structure

```
cadquery-web-ide/
├── main.py              # FastAPI backend server
├── index.html           # Frontend (Monaco + Three.js)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production container
├── docker-compose.yml   # Docker orchestration
├── .env.example         # Configuration template
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web application |
| `/render` | POST | Execute code, return GLB |
| `/export-stl` | POST | Execute code, return STL |
| `/health` | GET | Health check |
| `/docs` | GET | API docs (dev mode only) |

## Security

This application executes arbitrary Python code. For production deployments:

1. **Always use Docker** - Container isolation
2. **Restricted builtins** - Dangerous functions blocked (`open`, `exec`, `__import__`, etc.)
3. **Execution timeout** - Prevents infinite loops
4. **Rate limiting** - Prevents abuse
5. **Resource limits** - CPU/memory constraints via Docker

See [SECURITY.md](SECURITY.md) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE)

## Credits

Built by [tabibazar.com](https://tabibazar.com)

Powered by:
- [CadQuery](https://cadquery.readthedocs.io/) - Python CAD library
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Three.js](https://threejs.org/) - 3D visualization
- [Monaco Editor](https://microsoft.github.io/monaco-editor/) - Code editor
