"""GPU scene view: Gaussian splats drawn as depth-tested round point sprites
(OpenGL 3.3 core via Qt). Interactive at millions of splats; the CPU renderer
remains the fallback when OpenGL is unavailable."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QMatrix4x4, QSurfaceFormat
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_DEPTH_TEST = 0x0B71
GL_PROGRAM_POINT_SIZE = 0x8642
GL_POINTS = 0x0000
GL_FLOAT = 0x1406

VERT = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_col;
layout(location = 2) in float a_rad;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_focal;
uniform float u_scale;
out vec3 v_col;
void main() {
    vec4 p = u_view * vec4(a_pos, 1.0);
    gl_Position = u_proj * p;
    float z = max(-p.z, 1e-4);
    gl_PointSize = clamp(2.0 * u_scale * a_rad * u_focal / z, 1.0, 96.0);
    v_col = a_col;
}
"""

FRAG = """#version 330 core
in vec3 v_col;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;
    frag = vec4(v_col, 1.0);
}
"""


def gl_format() -> QSurfaceFormat:
    f = QSurfaceFormat()
    f.setVersion(3, 3)
    f.setProfile(QSurfaceFormat.CoreProfile)
    f.setDepthBufferSize(24)
    f.setSamples(4)
    return f


def splat_radii(scene) -> np.ndarray:
    """Isotropic footprint radius per splat (1.5 sigma of the two largest axes)."""
    s = np.sort(scene.scales.astype(np.float32), axis=1)
    return (1.5 * np.sqrt((s[:, 1] ** 2 + s[:, 2] ** 2) / 2)).astype(np.float32)


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def gl_view_matrix(c2w: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world → OpenGL view matrix (flip y and z)."""
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    return (np.diag([1.0, -1.0, -1.0, 1.0]) @ w2c).astype(np.float32)


class GLSplatView(QOpenGLWidget):
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(gl_format())
        self.program: QOpenGLShaderProgram | None = None
        self.vao: QOpenGLVertexArrayObject | None = None
        self.vbo: QOpenGLBuffer | None = None
        self.count = 0
        self.pending: np.ndarray | None = None
        self.c2w = np.eye(4)
        self.vfov = 50.0
        self.near, self.far = 0.01, 1000.0
        self.splat_scale = 1.0
        self.ok = False
        self.background = (0.094, 0.094, 0.11)

    # ---------------------------------------------------------------- data
    def set_points(self, means: np.ndarray, colors: np.ndarray, radii: np.ndarray) -> None:
        self.pending = np.ascontiguousarray(
            np.concatenate([means.astype(np.float32), np.clip(colors, 0, 1).astype(np.float32),
                            radii[:, None].astype(np.float32)], axis=1))
        self.update()

    def set_camera(self, c2w: np.ndarray, vfov_deg: float, near: float, far: float) -> None:
        self.c2w, self.vfov, self.near, self.far = c2w, vfov_deg, near, far
        self.update()

    # ---------------------------------------------------------------- GL
    def initializeGL(self):  # noqa: N802
        try:
            ctx = self.context()
            if ctx is None or not ctx.isValid() or ctx.format().majorVersion() < 3:
                raise RuntimeError("OpenGL 3.3 context not available")
            prog = QOpenGLShaderProgram(self)
            if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VERT):
                raise RuntimeError("vertex shader: " + prog.log())
            if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAG):
                raise RuntimeError("fragment shader: " + prog.log())
            if not prog.link():
                raise RuntimeError("link: " + prog.log())
            self.program = prog
            self.vao = QOpenGLVertexArrayObject(self)
            self.vao.create()
            self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self.vbo.create()
            self.ok = True
        except Exception as e:  # noqa: BLE001 - fall back to the CPU view
            self.ok = False
            self.failed.emit(str(e))

    def _upload(self) -> None:
        data = self.pending
        self.pending = None
        self.vao.bind()
        self.vbo.bind()
        self.vbo.allocate(data.tobytes(), data.nbytes)
        p = self.program
        p.bind()
        stride = 7 * 4
        p.enableAttributeArray(0)
        p.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride)
        p.enableAttributeArray(1)
        p.setAttributeBuffer(1, GL_FLOAT, 12, 3, stride)
        p.enableAttributeArray(2)
        p.setAttributeBuffer(2, GL_FLOAT, 24, 1, stride)
        self.vao.release()
        self.count = len(data)

    def paintGL(self):  # noqa: N802
        f = self.context().functions()
        f.glClearColor(*self.background, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.ok:
            return
        if self.pending is not None:
            self._upload()
        if not self.count:
            return
        f.glEnable(GL_DEPTH_TEST)
        f.glEnable(GL_PROGRAM_POINT_SIZE)
        dpr = self.devicePixelRatioF()
        h = max(1, int(self.height() * dpr))
        aspect = max(self.width(), 1) / max(self.height(), 1)
        proj = perspective(self.vfov, aspect, self.near, self.far)
        view = gl_view_matrix(self.c2w)
        p = self.program
        p.bind()
        # PySide6 resolves float uniforms only through the location-based setUniformValue1f
        p.setUniformValue(p.uniformLocation("u_view"), QMatrix4x4(*view.flatten().tolist()))
        p.setUniformValue(p.uniformLocation("u_proj"), QMatrix4x4(*proj.flatten().tolist()))
        p.setUniformValue1f(p.uniformLocation("u_focal"), float((h / 2) / np.tan(np.radians(self.vfov) / 2)))
        p.setUniformValue1f(p.uniformLocation("u_scale"), float(self.splat_scale))
        self.vao.bind()
        f.glDrawArrays(GL_POINTS, 0, self.count)
        self.vao.release()
