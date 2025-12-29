"""
CadQuery Web IDE

A browser-based IDE for CadQuery - the Python CAD modeling library.

GitHub: https://github.com/tabibazar/cadquery-web-ide
Author: tabibazar.com
License: MIT
"""

import os
import sys
import tempfile
import base64
import traceback
import logging
import signal
from typing import Optional
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import cadquery as cq
from cadquery import Assembly, Workplane, Shape, Compound

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Application configuration from environment variables."""
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    
    # Security settings
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))  # seconds
    MAX_CODE_LENGTH = int(os.getenv("MAX_CODE_LENGTH", "50000"))  # characters
    
    # Feature flags
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"
    
    @classmethod
    def is_production(cls) -> bool:
        return cls.APP_ENV == "production"


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging():
    """Configure logging based on environment."""
    log_level = logging.INFO if Config.is_production() else logging.DEBUG
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return logging.getLogger("cadquery-ide")

logger = setup_logging()

# =============================================================================
# Security Warning
# =============================================================================

SECURITY_WARNING = """
================================================================================
                        SECURITY NOTICE
================================================================================
This application executes arbitrary Python code submitted by users.

PRODUCTION DEPLOYMENT REQUIREMENTS:
1. Always run inside Docker container (provides sandboxing)
2. Use resource limits (CPU, memory, execution time)
3. Run as non-root user
4. Use HTTPS with proper certificates
5. Implement authentication if exposing publicly
6. Consider rate limiting for public deployments
7. Monitor logs for suspicious activity

The Docker container provides isolation, but additional security measures
are recommended for public-facing deployments.
================================================================================
"""

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="CadQuery Web IDE",
    description="Browser-based CAD modeling with CadQuery",
    version="1.0.0",
    docs_url="/docs" if not Config.is_production() else None,
    redoc_url="/redoc" if not Config.is_production() else None,
)

# Rate limiting - 10 renders per minute, 30 per hour per IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class RenderRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., max_length=Config.MAX_CODE_LENGTH)


class RenderResponse(BaseModel):
    """Response model for render endpoint."""
    success: bool
    glb_data: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: Optional[int] = None


class ExportResponse(BaseModel):
    """Response model for export endpoints."""
    success: bool
    data: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    environment: str
    cadquery_version: str
    timestamp: str


# =============================================================================
# Execution Timeout Handler
# =============================================================================

class ExecutionTimeout(Exception):
    """Raised when code execution exceeds timeout."""
    pass


@contextmanager
def execution_timeout(seconds: int):
    """Context manager for execution timeout (Unix only)."""
    def timeout_handler(signum, frame):
        raise ExecutionTimeout(f"Code execution exceeded {seconds} second timeout")
    
    # Only use signal-based timeout on Unix systems
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: no timeout enforcement (Docker handles this)
        yield


# =============================================================================
# Code Execution
# =============================================================================

def execute_user_code(code: str, namespace: dict) -> dict:
    """
    Execute user-provided CadQuery code.
    
    Security is provided by:
    - Docker container isolation
    - Execution timeout
    - Resource limits (via Docker)
    """
    compiled = compile(code, '<user_code>', 'exec')
    exec_func = getattr(__builtins__, 'exec', None) or __builtins__['exec']
    exec_func(compiled, namespace)
    return namespace


def create_safe_builtins() -> dict:
    """
    Create a restricted builtins dict that blocks dangerous operations.

    Blocked:
    - File operations: open, file
    - Code execution: exec, eval, compile, __import__
    - Process/system: exit, quit, input
    - Introspection that could be exploited: globals, locals, vars, dir, getattr, setattr, delattr
    - Memory manipulation: memoryview
    """
    # Safe builtins whitelist
    safe_names = [
        # Types
        'bool', 'int', 'float', 'complex', 'str', 'bytes', 'bytearray',
        'list', 'tuple', 'dict', 'set', 'frozenset',
        'type', 'object', 'slice', 'range',
        # Math
        'abs', 'divmod', 'pow', 'round', 'min', 'max', 'sum',
        # Iteration
        'len', 'iter', 'next', 'enumerate', 'zip', 'map', 'filter', 'reversed', 'sorted',
        # Boolean
        'all', 'any', 'True', 'False', 'None',
        # String/repr
        'repr', 'ascii', 'chr', 'ord', 'format', 'hash',
        'bin', 'hex', 'oct',
        # Object creation
        'callable', 'isinstance', 'issubclass', 'hasattr',
        'id', 'super',
        # Exceptions (needed for try/except)
        'Exception', 'BaseException', 'TypeError', 'ValueError', 'KeyError',
        'IndexError', 'AttributeError', 'RuntimeError', 'StopIteration',
        'ZeroDivisionError', 'OverflowError', 'NameError', 'AssertionError',
        # Print for debugging
        'print',
    ]

    import builtins
    safe_builtins = {}
    for name in safe_names:
        if hasattr(builtins, name):
            safe_builtins[name] = getattr(builtins, name)

    return safe_builtins


def create_namespace() -> dict:
    """Create execution namespace with CadQuery imports and restricted builtins."""
    return {
        "__builtins__": create_safe_builtins(),
        "cq": cq,
        "cadquery": cq,
        "Workplane": Workplane,
        "Assembly": Assembly,
        "Shape": Shape,
        "Compound": Compound,
        "Color": cq.Color,
        # Common imports users might need (pre-imported for safety)
        "math": __import__("math"),
    }


# =============================================================================
# Export Functions
# =============================================================================

def convert_to_glb(result) -> bytes:
    """Convert CadQuery result to GLB format."""
    if isinstance(result, Assembly):
        assy = result
    else:
        assy = Assembly()
        
        if isinstance(result, Workplane):
            solids = result.solids().vals()
            if solids:
                for i, solid in enumerate(solids):
                    assy.add(solid, name=f"solid_{i}", color=cq.Color(0.7, 0.7, 0.8))
            else:
                shapes = result.vals()
                for i, shape in enumerate(shapes):
                    if hasattr(shape, 'wrapped'):
                        assy.add(shape, name=f"shape_{i}", color=cq.Color(0.7, 0.7, 0.8))
        elif isinstance(result, (Shape, Compound)):
            assy.add(result, name="result", color=cq.Color(0.7, 0.7, 0.8))
        elif isinstance(result, (list, tuple)):
            for i, item in enumerate(result):
                if isinstance(item, Workplane):
                    solids = item.solids().vals()
                    for j, solid in enumerate(solids):
                        assy.add(solid, name=f"item_{i}_solid_{j}", color=cq.Color(0.7, 0.7, 0.8))
                elif isinstance(item, (Shape, Compound)):
                    assy.add(item, name=f"item_{i}", color=cq.Color(0.7, 0.7, 0.8))
        else:
            raise ValueError(f"Cannot convert {type(result).__name__} to GLB")

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        assy.export(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def convert_to_stl(result) -> bytes:
    """Convert CadQuery result to STL format."""
    if isinstance(result, Assembly):
        shapes = []
        for name, obj in result.traverse():
            if obj.obj is not None:
                shapes.append(obj.obj)
        if shapes:
            combined = shapes[0]
            for s in shapes[1:]:
                combined = combined.fuse(s)
            shape = combined
        else:
            raise ValueError("Assembly contains no shapes")
    elif isinstance(result, Workplane):
        shape = result.val()
    elif isinstance(result, (Shape, Compound)):
        shape = result
    else:
        raise ValueError(f"Cannot convert {type(result).__name__} to STL")

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        shape.exportStl(tmp_path, tolerance=0.01, angularTolerance=0.1)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/render", response_model=RenderResponse)
@limiter.limit("10/minute;30/hour")
async def render_code(request: Request, render_request: RenderRequest):
    """Execute CadQuery code and return GLB model."""
    start_time = datetime.now()
    code = render_request.code

    if not code or not code.strip():
        return RenderResponse(
            success=False,
            error="No code provided",
            error_type="ValidationError"
        )

    namespace = create_namespace()

    try:
        with execution_timeout(Config.EXECUTION_TIMEOUT):
            namespace = execute_user_code(code, namespace)

        if 'result' not in namespace:
            return RenderResponse(
                success=False,
                error="Code must assign the final object to a variable named 'result'.\n\n"
                      "Example:\n    result = cq.Workplane('XY').box(10, 20, 30)",
                error_type="ResultNotFound"
            )

        glb_bytes = convert_to_glb(namespace['result'])
        glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')
        
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"Render completed in {execution_time}ms")

        return RenderResponse(
            success=True,
            glb_data=glb_base64,
            execution_time_ms=execution_time
        )

    except ExecutionTimeout as e:
        logger.warning(f"Execution timeout: {str(e)}")
        return RenderResponse(
            success=False,
            error=str(e),
            error_type="TimeoutError"
        )
    except SyntaxError as e:
        return RenderResponse(
            success=False,
            error=f"Syntax Error at line {e.lineno}:\n{e.msg}\n\n{e.text or ''}",
            error_type="SyntaxError"
        )
    except Exception as e:
        logger.error(f"Render error: {type(e).__name__}: {str(e)}")
        tb = traceback.format_exc()
        return RenderResponse(
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n\n{tb}",
            error_type=type(e).__name__
        )


@app.post("/export-stl", response_model=ExportResponse)
@limiter.limit("5/minute;20/hour")
async def export_stl(request: Request, export_request: RenderRequest):
    """Export CadQuery result as STL file."""
    code = export_request.code

    if not code or not code.strip():
        return ExportResponse(success=False, error="No code provided", error_type="ValidationError")

    namespace = create_namespace()

    try:
        with execution_timeout(Config.EXECUTION_TIMEOUT):
            namespace = execute_user_code(code, namespace)

        if 'result' not in namespace:
            return ExportResponse(
                success=False,
                error="Code must assign the final object to 'result'",
                error_type="ResultNotFound"
            )

        stl_bytes = convert_to_stl(namespace['result'])
        stl_base64 = base64.b64encode(stl_bytes).decode('utf-8')

        return ExportResponse(
            success=True,
            data=stl_base64,
            filename="model.stl"
        )

    except Exception as e:
        logger.error(f"STL export error: {type(e).__name__}: {str(e)}")
        tb = traceback.format_exc()
        return ExportResponse(
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n\n{tb}",
            error_type=type(e).__name__
        )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend application."""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=404
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy",
        environment=Config.APP_ENV,
        cadquery_version=cq.__version__,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(SECURITY_WARNING)
    print(f"\n  Environment: {Config.APP_ENV}")
    print(f"  Server: http://{Config.APP_HOST}:{Config.APP_PORT}")
    print(f"  Execution timeout: {Config.EXECUTION_TIMEOUT}s")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=Config.APP_HOST,
        port=Config.APP_PORT,
        log_level="info" if Config.is_production() else "debug",
    )
