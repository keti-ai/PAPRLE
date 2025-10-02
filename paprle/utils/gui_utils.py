from PySide6.QtWidgets import QWidget, QToolButton, QApplication, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QMouseEvent, QPainter, QColor
import sys


class DraggableHandle(QToolButton):
    moved = Signal(int)  # Emit value position when moved

    def __init__(self, parent, value_callback):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.setCursor(Qt.OpenHandCursor)
        self.dragging = False
        self.value_callback = value_callback  # maps x -> value

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            global_x = self.mapToParent(event.position().toPoint()).x()
            value = self.value_callback(global_x)
            self.moved.emit(value)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self):
        super().__init__()
        #self.setMinimumHeight(60)
        self.min = 0
        self.max = 300
        self.start = 0
        self.end = 300
        self.track_height = 8
        self.handle_width = 14

        # Create handle buttons
        self.start_btn = DraggableHandle(self, self.pos_to_value)
        self.end_btn = DraggableHandle(self, self.pos_to_value)

        self.update_handle_positions()

    def setMinimum(self, value):
        self.min = value
        self.update_handle_positions()

    def setMaximum(self, value):
        self.max = value
        self.update_handle_positions()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        # Draw track
        track_y = h // 2 - self.track_height // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 200, 200))
        painter.drawRect(0, track_y, w, self.track_height)

        # Draw selected range
        x1 = self.value_to_pos(self.start)
        x2 = self.value_to_pos(self.end)
        painter.setBrush(QColor(100, 180, 255))
        painter.drawRect(x1, track_y, x2 - x1, self.track_height)

    def resizeEvent(self, event):
        self.update_handle_positions()

    def update_handle_positions(self):
        x1 = self.value_to_pos(self.start)
        x2 = self.value_to_pos(self.end)
        y = self.height() // 2 - self.handle_width // 2
        self.start_btn.move(x1 - self.handle_width // 2, y)
        self.end_btn.move(x2 - self.handle_width // 2, y)
        self.update()


    def update_start_end(self):
        self.update_handle_positions()
        self.rangeChanged.emit(self.start, self.end)


    def value_to_pos(self, val):
        span = self.max - self.min
        usable_width = self.width() - self.handle_width
        return int((val - self.min) / span * usable_width + self.handle_width / 2)

    def pos_to_value(self, x):
        span = self.max - self.min
        usable_width = self.width() - self.handle_width
        x = max(self.handle_width / 2, min(x, self.width() - self.handle_width / 2))
        return int((x - self.handle_width / 2) / usable_width * span + self.min)

    def set_start(self, value):
        self.start = min(value, self.end - 1)
        self.update_handle_positions()
        self.rangeChanged.emit(self.start, self.end)

    def set_end(self, value):
        self.end = max(value, self.start + 1)
        self.update_handle_positions()
        self.rangeChanged.emit(self.start, self.end)


# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    layout = QVBoxLayout(win)

    slider = RangeSlider()
    label = QLabel()
    slider.rangeChanged.connect(lambda s, e: label.setText(f"{s // 60}:{s % 60:02d} - {e // 60}:{e % 60:02d}"))

    layout.addWidget(slider)
    layout.addWidget(label)

    win.resize(400, 120)
    win.show()
    sys.exit(app.exec())
