"""
Color Cohesion Analyzer - Node Graph Canvas
Blueprint-style node visualization with pan/zoom
"""

import math
from typing import List, Dict, Optional, Tuple
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsDropShadowEffect,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMenu, QApplication
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QPainterPath, QLinearGradient, QRadialGradient, QTransform,
    QWheelEvent, QMouseEvent, QKeyEvent, QCursor
)

from .theme import COLORS, NODE_STYLE, CONNECTION_STYLE, get_cohesion_color, get_distance_line_style
from .translations import t
from core.palette_extraction import Palette
from core.color_space import PaletteColor


class PaletteSwatchItem(QGraphicsItem):
    """Individual color swatch in a palette node"""
    
    def __init__(self, color: PaletteColor, width: float, height: float, parent=None):
        super().__init__(parent)
        self.color = color
        self.width = width
        self.height = height
        self.show_hex = True
        self.hovered = False
        
        self.setAcceptHoverEvents(True)
        self._update_tooltip()
    
    def _update_tooltip(self):
        """Update tooltip text"""
        self.setToolTip(f"{self.color.hex_code}\n{t('click_to_copy')}")
    
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw color rectangle
        rect = self.boundingRect()
        painter.fillRect(rect, QColor(self.color.hex_code))
        
        # Draw hover highlight
        if self.hovered:
            painter.fillRect(rect, QColor("#FFFFFF33"))
        
        # Draw subtle border
        pen = QPen(QColor("#00000033"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect)
        
        # Draw hex code if enabled - always show, adjust font size based on width
        if self.show_hex:
            # Determine text color based on luminance
            r, g, b = self.color.rgb
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            text_color = "#000000" if luminance > 0.5 else "#FFFFFF"
            
            painter.setPen(QColor(text_color))
            
            # Adjust font size based on swatch width
            if self.width >= 35:
                font_size = 7
                hex_text = self.color.hex_code[1:].upper()  # Full hex without #
            elif self.width >= 25:
                font_size = 6
                hex_text = self.color.hex_code[1:4].upper()  # First 3 chars
            else:
                font_size = 5
                hex_text = self.color.hex_code[1:3].upper()  # First 2 chars
            
            font = QFont("Segoe UI", font_size)
            painter.setFont(font)
            
            text_rect = QRectF(1, self.height - 12, self.width - 2, 11)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, hex_text)
    
    def hoverEnterEvent(self, event):
        self.hovered = True
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.update()
    
    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update()
    
    def mousePressEvent(self, event):
        """Copy hex code to clipboard on click"""
        if event.button() == Qt.MouseButton.LeftButton:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.color.hex_code.upper())
            # Update tooltip temporarily
            self.setToolTip(f"{self.color.hex_code.upper()}\n{t('copied')}")
            # Reset tooltip after a moment
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1500, self._update_tooltip)
        super().mousePressEvent(event)


class PaletteNodeItem(QGraphicsItem):
    """
    Node representing a palette (asset, consensus, or aggregate)
    """
    
    # Signal-like callback for position changes
    position_changed_callback = None
    selection_callback = None  # Called when node is selected
    
    def __init__(
        self,
        palette: Palette,
        node_type: str = "default",
        width: float = 180,
        height: float = 120,
        parent=None
    ):
        super().__init__(parent)
        self.palette = palette
        self.node_type = node_type
        self.width = width
        self.height = height
        self.selected = False
        self.hovered = False
        self.expanded = False
        self.show_hex = True  # For display options
        
        # Visual settings
        self.style = NODE_STYLE.get(node_type, NODE_STYLE["default"])
        self.swatch_height = 40
        
        # Store swatch references for updates
        self.swatches: List[PaletteSwatchItem] = []
        
        # Enable interactions
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)  # For position tracking
        self.setAcceptHoverEvents(True)
        
        # Child items
        self._create_swatches()
    
    def itemChange(self, change, value):
        """Track position changes to update connections"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Notify parent view to update connections
            if self.position_changed_callback:
                self.position_changed_callback()
        return super().itemChange(change, value)
    
    def _create_swatches(self):
        """Create color swatch items"""
        if not self.palette.colors:
            return
        
        # Calculate swatch dimensions
        n_colors = min(len(self.palette.colors), 8)
        swatch_width = (self.width - 20) / n_colors
        swatch_y = self.height - self.swatch_height - 10
        
        self.swatches = []  # Clear existing
        for i, color in enumerate(self.palette.colors[:8]):
            swatch = PaletteSwatchItem(
                color, swatch_width, self.swatch_height, self
            )
            swatch.setPos(10 + i * swatch_width, swatch_y)
            self.swatches.append(swatch)
    
    def boundingRect(self) -> QRectF:
        margin = 4
        return QRectF(-margin, -margin, self.width + 2*margin, self.height + 2*margin)
    
    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get current style
        if self.selected:
            style = NODE_STYLE["selected"]
        elif self.hovered:
            style = {**self.style, "background": NODE_STYLE["default"]["background"]}
        else:
            style = self.style
        
        # Draw shadow for central nodes
        if self.node_type in ["consensus", "average"]:
            shadow_rect = QRectF(4, 4, self.width, self.height)
            painter.fillRect(shadow_rect, QColor("#00000044"))
        
        # Draw background
        rect = QRectF(0, 0, self.width, self.height)
        bg_color = QColor(style["background"])
        
        # Create rounded rect path
        path = QPainterPath()
        path.addRoundedRect(rect, style["border_radius"], style["border_radius"])
        
        painter.fillPath(path, QBrush(bg_color))
        
        # Draw border
        pen = QPen(QColor(style["border"]))
        pen.setWidth(style["border_width"])
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw header separator
        header_y = 35
        painter.setPen(QPen(QColor(style["border"]), 1))
        painter.drawLine(QPointF(10, header_y), QPointF(self.width - 10, header_y))
        
        # Draw title
        painter.setPen(QColor(style["text"]))
        title_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(title_font)
        
        # Truncate title if needed
        title = self.palette.source_name
        metrics = QFontMetrics(title_font)
        max_width = self.width - 20
        if metrics.horizontalAdvance(title) > max_width:
            title = metrics.elidedText(title, Qt.TextElideMode.ElideRight, int(max_width))
        
        painter.drawText(QRectF(10, 8, self.width - 20, 24), 
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                        title)
        
        # Draw type label
        type_font = QFont("Segoe UI", 8)
        painter.setFont(type_font)
        painter.setPen(QColor(COLORS["text_muted"]))
        
        type_label = self.palette.source_type.upper()
        if self.node_type == "consensus":
            type_label = "CONSENSUS"
        elif self.node_type == "average":
            type_label = "AVERAGE"
        elif self.node_type == "outlier":
            type_label = "OUTLIER"
        
        painter.drawText(QRectF(10, 40, self.width - 20, 16),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                        type_label)
        
        # Draw color count
        count_text = f"{len(self.palette.colors)} colors"
        painter.drawText(QRectF(10, 40, self.width - 20, 16),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                        count_text)
    
    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
    
    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected = not self.selected
            self.update()
            # Notify selection
            if self.selected and self.selection_callback:
                self.selection_callback(self.palette)
        super().mousePressEvent(event)


class ConnectionItem(QGraphicsPathItem):
    """
    Connection line between nodes showing palette relationship
    """
    
    def __init__(
        self,
        source_node: PaletteNodeItem,
        target_node: PaletteNodeItem,
        distance: float = 0,
        parent=None
    ):
        super().__init__(parent)
        self.source_node = source_node
        self.target_node = target_node
        self.distance = distance
        
        # Style based on distance
        self.line_style = get_distance_line_style(distance)
        self._update_path()
        
        self.setZValue(-1)  # Behind nodes
    
    def _update_path(self):
        """Update the path between nodes"""
        # Get center points
        source_rect = self.source_node.boundingRect()
        target_rect = self.target_node.boundingRect()
        
        source_center = self.source_node.scenePos() + source_rect.center()
        target_center = self.target_node.scenePos() + target_rect.center()
        
        # Create curved path
        path = QPainterPath()
        path.moveTo(source_center)
        
        # Control points for bezier curve
        dx = target_center.x() - source_center.x()
        dy = target_center.y() - source_center.y()
        
        ctrl1 = QPointF(source_center.x() + dx * 0.3, source_center.y())
        ctrl2 = QPointF(target_center.x() - dx * 0.3, target_center.y())
        
        path.cubicTo(ctrl1, ctrl2, target_center)
        
        self.setPath(path)
        
        # Style
        color = QColor(self.line_style["color"])
        pen = QPen(color)
        pen.setWidth(self.line_style["width"])
        
        if self.line_style["style"] == "dashed":
            pen.setStyle(Qt.PenStyle.DashLine)
        
        self.setPen(pen)
    
    def update_position(self):
        """Called when nodes move"""
        self._update_path()


class NodeGraphView(QGraphicsView):
    """
    Main canvas for the node graph with pan/zoom support
    Blueprint-style grid background
    """
    
    node_selected = pyqtSignal(object)  # Emits Palette
    node_double_clicked = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Style
        self.setStyleSheet(f"background: {COLORS['bg_primary']}; border: none;")
        
        # Zoom/pan state
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 3.0
        self.panning = False
        self.pan_start = QPointF()
        
        # Grid settings
        self.grid_size = 50
        self.show_grid = True
        
        # Node storage
        self.nodes: Dict[str, PaletteNodeItem] = {}
        self.connections: List[ConnectionItem] = []
        
        # Layout
        self.center_pos = QPointF(0, 0)
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw blueprint-style grid background"""
        super().drawBackground(painter, rect)
        
        if not self.show_grid:
            return
        
        # Fill background
        painter.fillRect(rect, QColor(COLORS["bg_primary"]))
        
        # Calculate grid bounds
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)
        
        # Draw minor grid
        minor_pen = QPen(QColor(COLORS["grid_primary"]))
        minor_pen.setWidth(1)
        painter.setPen(minor_pen)
        
        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += self.grid_size
        
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += self.grid_size
        
        # Draw major grid (every 5 cells)
        major_pen = QPen(QColor(COLORS["grid_secondary"]))
        major_pen.setWidth(1)
        painter.setPen(major_pen)
        
        major_size = self.grid_size * 5
        x = left - (left % major_size)
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += major_size
        
        y = top - (top % major_size)
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += major_size
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel"""
        delta = event.angleDelta().y()
        
        if delta > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        
        new_zoom = self.zoom_level * factor
        
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom
            self.scale(factor, factor)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle pan start - middle button or right button"""
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.panning = True
            self.pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle panning"""
        if self.panning:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle pan end"""
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Home:
            self.reset_view()
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        else:
            super().keyPressEvent(event)
    
    def zoom_in(self):
        """Zoom in"""
        if self.zoom_level < self.max_zoom:
            factor = 1.15
            self.zoom_level *= factor
            self.scale(factor, factor)
    
    def zoom_out(self):
        """Zoom out"""
        if self.zoom_level > self.min_zoom:
            factor = 1 / 1.15
            self.zoom_level *= factor
            self.scale(factor, factor)
    
    def reset_view(self):
        """Reset to default view"""
        self.resetTransform()
        self.zoom_level = 1.0
        self.centerOn(self.center_pos)
    
    def fit_to_content(self):
        """Fit view to show all nodes"""
        if self.nodes:
            self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            # Update zoom level
            self.zoom_level = self.transform().m11()
    
    def add_node(
        self,
        palette: Palette,
        node_type: str = "default",
        position: Optional[QPointF] = None
    ) -> PaletteNodeItem:
        """Add a palette node to the graph"""
        # Determine size based on type - larger sizes for better hex visibility
        if node_type in ["consensus", "average"]:
            width, height = 320, 180
        else:
            width, height = 280, 140
        
        node = PaletteNodeItem(palette, node_type, width, height)
        
        # Set callback to update connections when node moves
        node.position_changed_callback = self._update_all_connections
        # Set callback for selection
        node.selection_callback = self._on_node_selected
        
        if position:
            node.setPos(position)
        
        self._scene.addItem(node)
        self.nodes[palette.source_name] = node
        
        return node
    
    def add_connection(
        self,
        source_name: str,
        target_name: str,
        distance: float = 0
    ) -> Optional[ConnectionItem]:
        """Add connection between nodes"""
        if source_name not in self.nodes or target_name not in self.nodes:
            return None
        
        connection = ConnectionItem(
            self.nodes[source_name],
            self.nodes[target_name],
            distance
        )
        
        self._scene.addItem(connection)
        self.connections.append(connection)
        
        return connection
    
    def clear_graph(self):
        """Clear all nodes and connections"""
        self._scene.clear()
        self.nodes.clear()
        self.connections.clear()
    
    def layout_radial(self, center_names: List[str]):
        """
        Layout nodes in radial pattern
        Center nodes in middle, others arranged around them
        """
        # Position center nodes
        center_spacing = 300
        center_y = 0
        
        for i, name in enumerate(center_names):
            if name in self.nodes:
                x = (i - len(center_names) / 2) * center_spacing
                self.nodes[name].setPos(x, center_y)
        
        # Position peripheral nodes in a circle
        peripheral = [n for n in self.nodes.keys() if n not in center_names]
        n_peripheral = len(peripheral)
        
        if n_peripheral > 0:
            radius = 400
            angle_step = 2 * math.pi / n_peripheral
            
            for i, name in enumerate(peripheral):
                angle = i * angle_step - math.pi / 2
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) + 100  # Offset below centers
                self.nodes[name].setPos(x, y)
        
        # Update connections
        self._update_all_connections()
        
        # Fit view
        self.fit_to_content()
    
    def layout_grid(self, columns: int = 4):
        """Layout nodes in a grid pattern"""
        spacing_x = 220
        spacing_y = 150
        
        for i, (name, node) in enumerate(self.nodes.items()):
            row = i // columns
            col = i % columns
            node.setPos(col * spacing_x, row * spacing_y)
        
        self._update_all_connections()
        self.fit_to_content()
    
    def _update_all_connections(self):
        """Update all connection paths - called when nodes move"""
        for conn in self.connections:
            conn.update_position()
    
    def _on_node_selected(self, palette):
        """Handle node selection - emit signal"""
        self.node_selected.emit(palette)
    
    def set_show_connections(self, show: bool):
        """Show or hide connection lines"""
        for conn in self.connections:
            conn.setVisible(show)
        self._scene.update()
        self.viewport().update()
    
    def set_show_hex(self, show: bool):
        """Show or hide hex codes on swatches"""
        for node in self.nodes.values():
            node.show_hex = show
            # Update all swatches in the node
            for swatch in node.swatches:
                swatch.show_hex = show
                swatch.update()
            node.update()
        self._scene.update()
        self.viewport().update()
    
    def set_compact_mode(self, compact: bool):
        """Toggle compact node display"""
        scale = 0.7 if compact else 1.0
        for node in self.nodes.values():
            node.setScale(scale)
        self._update_all_connections()
