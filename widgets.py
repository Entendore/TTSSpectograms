# widgets.py
import os
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QAbstractItemView, QMenu, QHeaderView,
    QWidget, QHBoxLayout, QSlider, QLabel, QSizePolicy, QToolButton,
    QVBoxLayout, QFrame,
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush


class AudioTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Filename", "Duration", "Status"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        
        self.setShowGrid(False)
        self.setStyleSheet("""
            QTableWidget {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #1e1e1e;
                alternate-background-color: #242424;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #2ecc71;
                color: black;
            }
        """)

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; }
            QMenu::item:selected { background-color: #3c3c3c; }
        """)
        
        remove_action = menu.addAction("Remove Selected")
        menu.addSeparator()
        open_folder_action = menu.addAction("Open Containing Folder")
        
        action = menu.exec_(self.mapToGlobal(pos))
        
        if action == remove_action:
            self._remove_selected_items()
        elif action == open_folder_action:
            filepath = self.item(item.row(), 0).data(Qt.UserRole)
            if filepath:
                from utils import open_file_location
                open_file_location(filepath)

    def _remove_selected_items(self):
        rows = set()
        for item in self.selectedItems():
            rows.add(item.row())
        
        for row in sorted(rows, reverse=True):
            self.removeRow(row)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if local_path:
                    paths.append(local_path)
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()


class LabeledSlider(QWidget):
    valueChanged = Signal(int)
    sliderReleased = Signal()

    def __init__(self, minimum=0, maximum=100, default_value=0, 
                 prefix="", suffix="", orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self._prefix = prefix
        self._suffix = suffix

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)

        self.slider = QSlider(orientation)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(default_value)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(max(1, (maximum - minimum) // 10))

        self.label = QLabel(self._format_text(default_value))
        self.label.setMinimumWidth(45)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #5dade2; font-weight: bold;")

        self.layout.addWidget(self.slider, 1)
        self.layout.addWidget(self.label)

        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.sliderReleased.connect(self.sliderReleased.emit)

    def _format_text(self, value):
        text = f"{self._prefix}{value}{self._suffix}"
        if self._prefix == "+" and value >= 0:
            text = f"+{value}{self._suffix}"
        elif self._prefix == "+" and value < 0:
            text = f"{value}{self._suffix}"
        return text

    def _on_value_changed(self, value):
        self.label.setText(self._format_text(value))
        self.valueChanged.emit(value)

    def value(self):
        return self.slider.value()

    def setValue(self, val):
        self.slider.setValue(val)

    def setRange(self, min_val, max_val):
        self.slider.setRange(min_val, max_val)

    def setPrefix(self, prefix):
        self._prefix = prefix
        self.label.setText(self._format_text(self.slider.value()))

    def setSuffix(self, suffix):
        self._suffix = suffix
        self.label.setText(self._format_text(self.slider.value()))


class PalettePreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setMinimumWidth(180)
        self._pixmap = QPixmap()
        self.setStyleSheet("border: 1px solid #555; border-radius: 3px;")

    def set_lut(self, lut_data: np.ndarray):
        if lut_data is None or lut_data.shape[0] < 1:
            self._pixmap = QPixmap()
            self.update()
            return

        try:
            grad = np.arange(256, dtype=np.uint8).reshape(1, 256)
            colored_bgr = cv2.applyColorMap(grad, lut_data)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            
            h, w, ch = colored_rgb.shape
            qimg = QImage(colored_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            
            self._pixmap = QPixmap.fromImage(qimg).scaled(
                self.width(), self.height(), 
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            self.update()
        except Exception as e:
            print(f"Error generating palette preview: {e}")
            self._pixmap = QPixmap()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self._pixmap.isNull():
            painter.drawPixmap(0, 0, self.width(), self.height(), self._pixmap)
        else:
            painter.setBrush(QBrush(QColor(40, 40, 40)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, 0, self.width(), self.height())
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "(no palette)")

    def resizeEvent(self, event):
        super().resizeEvent(event)


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self._toggle_button = QToolButton()
        self._toggle_button.setStyleSheet("""
            QToolButton { 
                border: 1px solid #3a3a3a; 
                border-radius: 3px; 
                padding: 6px; 
                background-color: #2d2d2d; 
                color: #e0e0e0; 
                font-weight: bold; 
                text-align: left;
            }
            QToolButton:hover { background-color: #3c3c3c; }
            QToolButton:checked { background-color: #34495e; }
        """)
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.RightArrow)
        self._toggle_button.setText(f"  {title}")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)

        self._content_area = QWidget()
        self._content_area.setMaximumHeight(0)
        self._content_area.setMinimumHeight(0)
        self._content_area.setVisible(False)

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 4)
        self._layout.addWidget(self._toggle_button)
        self._layout.addWidget(self._content_area)

        self._toggle_button.clicked.connect(self._on_toggle)

    def _on_toggle(self, checked):
        if checked:
            self._toggle_button.setArrowType(Qt.DownArrow)
            self._content_area.setVisible(True)
            self._content_area.setMaximumHeight(16777215)
            self._content_area.setMinimumHeight(0)
        else:
            self._toggle_button.setArrowType(Qt.RightArrow)
            self._content_area.setMaximumHeight(0)
            QTimer.singleShot(200, lambda: self._content_area.setVisible(False) if not self._toggle_button.isChecked() else None)

    def content_layout(self) -> QVBoxLayout:
        if self._content_area.layout() is None:
            layout = QVBoxLayout(self._content_area)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)
        return self._content_area.layout()

    def set_content_layout(self, layout: QVBoxLayout):
        self._content_area.setLayout(layout)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
    def is_expanded(self) -> bool:
        return self._toggle_button.isChecked()

    def set_expanded(self, expand: bool):
        self._toggle_button.setChecked(expand)
        self._on_toggle(expand)