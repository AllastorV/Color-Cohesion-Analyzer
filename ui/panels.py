"""
Color Cohesion Analyzer - Sidebar Panels
Metrics display, filters, and controls
"""

from typing import Optional, List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QPushButton, QComboBox, QCheckBox,
    QSlider, QGroupBox, QProgressBar, QTreeWidget,
    QTreeWidgetItem, QSplitter, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

from .theme import COLORS, get_cohesion_color
from .translations import t, add_language_change_listener

# Lazy imports to avoid circular dependencies
def _get_palette_class():
    from core.palette_extraction import Palette
    return Palette

def _get_metrics_classes():
    from core.metrics import AssetMetrics, ProjectMetrics
    return AssetMetrics, ProjectMetrics


class MetricCard(QFrame):
    """Card displaying a single metric value"""
    
    def __init__(self, label: str, value: str = "-", color: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metric")
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)
        
        # Label
        self.name_label = QLabel(label)
        self.name_label.setObjectName("metricLabel")
        layout.addWidget(self.name_label)
    
    def set_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


class PalettePreview(QFrame):
    """Preview widget showing palette colors"""
    
    color_clicked = pyqtSignal(str)  # Emits hex code
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(60)
        self.palette = None  # Will hold Palette object
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        
        # Title
        self.title_label = QLabel("Palette")
        self.title_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.layout.addWidget(self.title_label)
        
        # Colors container
        self.colors_widget = QWidget()
        self.colors_layout = QHBoxLayout(self.colors_widget)
        self.colors_layout.setContentsMargins(0, 0, 0, 0)
        self.colors_layout.setSpacing(4)
        self.layout.addWidget(self.colors_widget)
    
    def set_palette(self, palette, title: str = None):
        """Set palette to display"""
        self.palette = palette
        
        if title:
            self.title_label.setText(title)
        
        # Clear existing colors
        while self.colors_layout.count():
            item = self.colors_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not palette or not palette.colors:
            return
        
        # Add color swatches
        for color in palette.colors[:8]:
            swatch = QFrame()
            swatch.setFixedSize(32, 32)
            swatch.setStyleSheet(
                f"background-color: {color.hex_code}; "
                f"border-radius: 4px; "
                f"border: 1px solid {COLORS['grid_accent']};"
            )
            swatch.setToolTip(f"{color.hex_code}\n{color.dominance*100:.1f}%")
            self.colors_layout.addWidget(swatch)
        
        self.colors_layout.addStretch()


class AssetMetricsPanel(QFrame):
    """Panel showing metrics for selected asset"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        self.header = QLabel(t("panel_asset_metrics"))
        self.header.setObjectName("header")
        layout.addWidget(self.header)
        
        # Palette preview
        self.palette_preview = PalettePreview()
        layout.addWidget(self.palette_preview)
        
        # Metrics grid
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(8)
        
        self.cohesion_card = MetricCard(t("label_cohesion"), "-")
        self.cohesion_card.setToolTip(t("tooltip_cohesion"))
        self.entropy_card = MetricCard(t("label_entropy"), "-")
        self.entropy_card.setToolTip(t("tooltip_entropy"))
        self.distance_card = MetricCard(t("label_distance"), "-")
        self.distance_card.setToolTip(t("tooltip_distance"))
        
        metrics_layout.addWidget(self.cohesion_card)
        metrics_layout.addWidget(self.entropy_card)
        metrics_layout.addWidget(self.distance_card)
        
        layout.addWidget(metrics_widget)
        
        # Additional metrics
        self.warm_cool_label = QLabel(f"{t('label_temperature')} -")
        self.warm_cool_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.warm_cool_label.setToolTip(t("tooltip_temperature"))
        layout.addWidget(self.warm_cool_label)
        
        self.saturation_label = QLabel(f"{t('label_saturation')} -")
        self.saturation_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.saturation_label.setToolTip(t("tooltip_saturation"))
        layout.addWidget(self.saturation_label)
        
        # Divergent colors
        self.divergent_label = QLabel(f"{t('label_divergent')} {t('label_none')}")
        self.divergent_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.divergent_label.setToolTip(t("tooltip_divergent"))
        self.divergent_label.setWordWrap(True)
        layout.addWidget(self.divergent_label)
        
        layout.addStretch()
        
        # Register for language changes
        add_language_change_listener(self._update_texts)
    
    def _update_texts(self):
        """Update texts when language changes"""
        self.header.setText(t("panel_asset_metrics"))
        self.cohesion_card.name_label.setText(t("label_cohesion"))
        self.cohesion_card.setToolTip(t("tooltip_cohesion"))
        self.entropy_card.name_label.setText(t("label_entropy"))
        self.entropy_card.setToolTip(t("tooltip_entropy"))
        self.distance_card.name_label.setText(t("label_distance"))
        self.distance_card.setToolTip(t("tooltip_distance"))
        self.warm_cool_label.setToolTip(t("tooltip_temperature"))
        self.saturation_label.setToolTip(t("tooltip_saturation"))
        self.divergent_label.setToolTip(t("tooltip_divergent"))
    
    def set_metrics(self, palette, metrics):
        """Update display with asset metrics
        
        Args:
            palette: Palette object
            metrics: AssetMetrics object
        """
        self.palette_preview.set_palette(palette, palette.source_name)
        
        # Cohesion score
        cohesion_color = get_cohesion_color(metrics.cohesion_score)
        self.cohesion_card.set_value(f"{metrics.cohesion_score:.2f}", cohesion_color)
        
        # Entropy
        self.entropy_card.set_value(f"{metrics.entropy:.2f}")
        
        # Distance
        self.distance_card.set_value(f"{metrics.distance_to_consensus:.1f}")
        
        # Temperature - convert warm/cool balance to Kelvin approximation
        # Warm (+1) ≈ 2700K, Neutral (0) ≈ 5500K, Cool (-1) ≈ 9000K
        kelvin = int(5500 - (metrics.warm_cool_balance * 2800))
        kelvin = max(2000, min(10000, kelvin))  # Clamp to realistic range
        
        if metrics.warm_cool_balance > 0.1:
            temp = f"{t('label_warm')} ({kelvin}K)"
        elif metrics.warm_cool_balance < -0.1:
            temp = f"{t('label_cool')} ({kelvin}K)"
        else:
            temp = f"{t('label_neutral')} ({kelvin}K)"
        self.warm_cool_label.setText(f"{t('label_temperature')} {temp}")
        
        # Saturation
        self.saturation_label.setText(
            f"{t('label_saturation')} Mean {metrics.saturation_mean:.1f}, "
            f"Std {metrics.saturation_std:.1f}"
        )
        
        # Divergent colors
        if metrics.divergent_colors:
            colors_str = ", ".join(metrics.divergent_colors[:5])
            if len(metrics.divergent_colors) > 5:
                colors_str += f" (+{len(metrics.divergent_colors) - 5} more)"
            self.divergent_label.setText(f"{t('label_divergent')} {colors_str}")
        else:
            self.divergent_label.setText(f"{t('label_divergent')} {t('label_none')}")
    
    def clear(self):
        """Clear the display"""
        self.palette_preview.set_palette(None)
        self.cohesion_card.set_value("-")
        self.entropy_card.set_value("-")
        self.distance_card.set_value("-")
        self.warm_cool_label.setText(f"{t('label_temperature')} -")
        self.saturation_label.setText(f"{t('label_saturation')} -")
        self.divergent_label.setText(f"{t('label_divergent')} {t('label_none')}")


class ProjectOverviewPanel(QFrame):
    """Panel showing overall project metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        self.header = QLabel(t("panel_project_overview"))
        self.header.setObjectName("header")
        layout.addWidget(self.header)
        
        # Asset counts
        self.assets_label = QLabel(f"{t('label_assets')} -")
        layout.addWidget(self.assets_label)
        
        # Cohesion metrics
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(8)
        
        self.avg_cohesion_card = MetricCard(t("label_avg_cohesion"), "-")
        self.avg_cohesion_card.setToolTip(t("tooltip_avg_cohesion"))
        self.outliers_card = MetricCard(t("label_outliers"), "-")
        self.outliers_card.setToolTip(t("tooltip_outliers"))
        
        metrics_layout.addWidget(self.avg_cohesion_card)
        metrics_layout.addWidget(self.outliers_card)
        
        layout.addWidget(metrics_widget)
        
        # Temperature distribution
        self.temp_group = QGroupBox(t("label_temp_distribution"))
        temp_layout = QVBoxLayout(self.temp_group)
        
        self.warm_bar = QProgressBar()
        self.warm_bar.setFormat(f"{t('label_warm_bar')} %v%")
        self.warm_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {COLORS['warning']}; }}")
        temp_layout.addWidget(self.warm_bar)
        
        self.cool_bar = QProgressBar()
        self.cool_bar.setFormat(f"{t('label_cool_bar')} %v%")
        self.cool_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {COLORS['accent_cyan']}; }}")
        temp_layout.addWidget(self.cool_bar)
        
        layout.addWidget(self.temp_group)
        
        # Recommendations
        self.recommendations_label = QLabel(f"{t('label_recommendations')}")
        self.recommendations_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.recommendations_label.setWordWrap(True)
        layout.addWidget(self.recommendations_label)
        
        layout.addStretch()
        
        # Register for language changes
        add_language_change_listener(self._update_texts)
    
    def _update_texts(self):
        """Update texts when language changes"""
        self.header.setText(t("panel_project_overview"))
        self.avg_cohesion_card.name_label.setText(t("label_avg_cohesion"))
        self.avg_cohesion_card.setToolTip(t("tooltip_avg_cohesion"))
        self.outliers_card.name_label.setText(t("label_outliers"))
        self.outliers_card.setToolTip(t("tooltip_outliers"))
        self.temp_group.setTitle(t("label_temp_distribution"))
    
    def set_project_metrics(self, metrics, recommendations: List[str] = None):
        """Update display with project metrics
        
        Args:
            metrics: ProjectMetrics object
            recommendations: Optional list of recommendation strings
        """
        # Asset counts
        assets_text = f"{t('label_assets')} {metrics.total_assets} ("
        parts = []
        if metrics.total_images > 0:
            parts.append(f"{metrics.total_images} {t('label_images')}")
        if metrics.total_videos > 0:
            parts.append(f"{metrics.total_videos} {t('label_videos')}")
        if metrics.total_shots > 0:
            parts.append(f"{metrics.total_shots} {t('label_shots')}")
        assets_text += ", ".join(parts) + ")"
        self.assets_label.setText(assets_text)
        
        # Cohesion
        cohesion_color = get_cohesion_color(metrics.average_cohesion)
        self.avg_cohesion_card.set_value(f"{metrics.average_cohesion:.2f}", cohesion_color)
        
        # Outliers
        outlier_color = COLORS["error"] if metrics.outlier_percentage > 30 else COLORS["text_primary"]
        self.outliers_card.set_value(
            f"{metrics.outlier_count} ({metrics.outlier_percentage:.0f}%)",
            outlier_color
        )
        
        # Temperature - warm_bias/cool_bias are 0-1 values
        # Scale to percentage (0-100)
        warm_pct = int(abs(metrics.warm_bias) * 100) if hasattr(metrics, 'warm_bias') and metrics.warm_bias else 0
        cool_pct = int(abs(metrics.cool_bias) * 100) if hasattr(metrics, 'cool_bias') and metrics.cool_bias else 0
        
        # Ensure valid range
        self.warm_bar.setValue(max(0, min(warm_pct, 100)))
        self.warm_bar.setFormat(f"{t('label_warm_bar')} {warm_pct}%")
        self.cool_bar.setValue(max(0, min(cool_pct, 100)))
        self.cool_bar.setFormat(f"{t('label_cool_bar')} {cool_pct}%")
        
        # Recommendations
        if recommendations:
            rec_text = f"{t('label_recommendations')}\n• " + "\n• ".join(recommendations[:3])
            self.recommendations_label.setText(rec_text)
        else:
            self.recommendations_label.setText(f"{t('label_recommendations')} {t('label_no_issues')}")


class FilterPanel(QFrame):
    """Panel for filtering the node graph"""
    
    filter_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        self.header = QLabel(t("panel_filters"))
        self.header.setObjectName("header")
        layout.addWidget(self.header)
        
        # View mode
        self.view_label = QLabel(t("label_view_mode"))
        self.view_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(self.view_label)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems([t("view_all_assets"), t("view_images_only"), t("view_videos_only"), t("view_outliers_only")])
        self.view_combo.setToolTip(t("tooltip_view_mode"))
        self.view_combo.currentIndexChanged.connect(self.filter_changed)
        layout.addWidget(self.view_combo)
        
        # Reference mode
        self.reference_check = QCheckBox(t("label_reference_mode"))
        self.reference_check.setToolTip(t("tooltip_reference_mode"))
        self.reference_check.stateChanged.connect(self.filter_changed)
        layout.addWidget(self.reference_check)
        
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip(t("tooltip_reference_select"))
        self.reference_combo.setEnabled(False)
        self.reference_check.stateChanged.connect(
            lambda state: self.reference_combo.setEnabled(state == Qt.CheckState.Checked.value)
        )
        layout.addWidget(self.reference_combo)
        
        # Display options
        self.display_group = QGroupBox(t("label_display_options"))
        display_layout = QVBoxLayout(self.display_group)
        
        self.show_hex_check = QCheckBox(t("label_show_hex"))
        self.show_hex_check.setToolTip(t("tooltip_show_hex"))
        self.show_hex_check.setChecked(True)
        self.show_hex_check.stateChanged.connect(self.filter_changed)
        display_layout.addWidget(self.show_hex_check)
        
        self.show_connections_check = QCheckBox(t("label_show_connections"))
        self.show_connections_check.setToolTip(t("tooltip_show_connections"))
        self.show_connections_check.setChecked(True)
        self.show_connections_check.stateChanged.connect(self.filter_changed)
        display_layout.addWidget(self.show_connections_check)
        
        self.compact_check = QCheckBox(t("label_compact_mode"))
        self.compact_check.setToolTip(t("tooltip_compact_mode"))
        self.compact_check.stateChanged.connect(self.filter_changed)
        display_layout.addWidget(self.compact_check)
        
        layout.addWidget(self.display_group)
        
        # Threshold slider
        self.threshold_label = QLabel(t("label_distance_threshold"))
        self.threshold_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(5, 50)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.filter_changed)
        layout.addWidget(self.threshold_slider)
        
        self.threshold_value = QLabel("10 ΔE")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value.setText(f"{v} ΔE")
        )
        layout.addWidget(self.threshold_value)
        
        layout.addStretch()
        
        # Register for language changes
        add_language_change_listener(self._update_texts)
    
    def _update_texts(self):
        """Update texts when language changes"""
        self.header.setText(t("panel_filters"))
        self.view_label.setText(t("label_view_mode"))
        
        # Update combo box items
        current_idx = self.view_combo.currentIndex()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItems([t("view_all_assets"), t("view_images_only"), t("view_videos_only"), t("view_outliers_only")])
        self.view_combo.setCurrentIndex(current_idx)
        self.view_combo.blockSignals(False)
        self.view_combo.setToolTip(t("tooltip_view_mode"))
        
        self.reference_check.setText(t("label_reference_mode"))
        self.reference_check.setToolTip(t("tooltip_reference_mode"))
        self.reference_combo.setToolTip(t("tooltip_reference_select"))
        self.display_group.setTitle(t("label_display_options"))
        self.show_hex_check.setText(t("label_show_hex"))
        self.show_hex_check.setToolTip(t("tooltip_show_hex"))
        self.show_connections_check.setText(t("label_show_connections"))
        self.show_connections_check.setToolTip(t("tooltip_show_connections"))
        self.compact_check.setText(t("label_compact_mode"))
        self.compact_check.setToolTip(t("tooltip_compact_mode"))
        self.threshold_label.setText(t("label_distance_threshold"))
    
    def set_reference_options(self, asset_names: List[str]):
        """Set available reference options"""
        self.reference_combo.clear()
        self.reference_combo.addItems(asset_names)
    
    def get_filter_state(self) -> Dict:
        """Get current filter settings"""
        return {
            "view_mode": self.view_combo.currentText(),
            "reference_mode": self.reference_check.isChecked(),
            "reference_asset": self.reference_combo.currentText(),
            "show_hex": self.show_hex_check.isChecked(),
            "show_connections": self.show_connections_check.isChecked(),
            "compact": self.compact_check.isChecked(),
            "threshold": self.threshold_slider.value()
        }


class AssetTreePanel(QFrame):
    """Tree view of all assets"""
    
    asset_selected = pyqtSignal(str)  # Emits asset name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 12, 16, 12)
        
        self.header = QLabel(t("panel_assets"))
        self.header.setObjectName("header")
        header_layout.addWidget(self.header)
        
        self.count_label = QLabel(f"0 {t('label_items')}")
        self.count_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        header_layout.addWidget(self.count_label)
        
        header_layout.addStretch()
        
        layout.addWidget(header_widget)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)
        
        # Register for language changes
        add_language_change_listener(self._update_texts)
    
    def _update_texts(self):
        """Update texts when language changes"""
        self.header.setText(t("panel_assets"))
    
    def set_assets(
        self,
        images: List[str],
        videos: List[Dict],
        consensus_name: str = None,
        average_name: str = None
    ):
        """Populate the tree with assets"""
        self.tree.clear()
        
        # Use translated names if not provided
        if consensus_name is None:
            consensus_name = t("label_consensus")
        if average_name is None:
            average_name = t("label_global_average")
        
        total_count = len(images) + len(videos)
        
        # Central palettes
        central_item = QTreeWidgetItem([t("label_central_palettes")])
        central_item.setExpanded(True)
        
        consensus_item = QTreeWidgetItem([consensus_name])
        consensus_item.setData(0, Qt.ItemDataRole.UserRole, consensus_name)
        central_item.addChild(consensus_item)
        
        average_item = QTreeWidgetItem([average_name])
        average_item.setData(0, Qt.ItemDataRole.UserRole, average_name)
        central_item.addChild(average_item)
        
        self.tree.addTopLevelItem(central_item)
        
        # Images
        if images:
            images_item = QTreeWidgetItem([f"{t('label_images').capitalize()} ({len(images)})"])
            images_item.setExpanded(True)
            
            for img_name in images:
                item = QTreeWidgetItem([img_name])
                item.setData(0, Qt.ItemDataRole.UserRole, img_name)
                images_item.addChild(item)
            
            self.tree.addTopLevelItem(images_item)
        
        # Videos
        if videos:
            videos_item = QTreeWidgetItem([f"{t('label_videos').capitalize()} ({len(videos)})"])
            videos_item.setExpanded(True)
            
            for video in videos:
                video_item = QTreeWidgetItem([video["name"]])
                video_item.setData(0, Qt.ItemDataRole.UserRole, video["name"])
                
                # Add shots as children
                if "shots" in video:
                    for shot in video["shots"]:
                        shot_item = QTreeWidgetItem([shot["name"]])
                        shot_item.setData(0, Qt.ItemDataRole.UserRole, shot["name"])
                        video_item.addChild(shot_item)
                
                videos_item.addChild(video_item)
            
            self.tree.addTopLevelItem(videos_item)
        
        self.count_label.setText(f"{total_count} {t('label_items')}")
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click"""
        asset_name = item.data(0, Qt.ItemDataRole.UserRole)
        if asset_name:
            self.asset_selected.emit(asset_name)
    
    def select_asset(self, asset_name: str):
        """Select an asset in the tree by name"""
        def find_item(parent_item, name):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.data(0, Qt.ItemDataRole.UserRole) == name:
                    return child
                # Search children recursively
                found = find_item(child, name)
                if found:
                    return found
            return None
        
        # Search all top level items
        for i in range(self.tree.topLevelItemCount()):
            top_item = self.tree.topLevelItem(i)
            found = find_item(top_item, asset_name)
            if found:
                self.tree.setCurrentItem(found)
                return


class SidebarPanel(QWidget):
    """Main sidebar containing all panels"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Scroll area for panels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        panels_widget = QWidget()
        panels_layout = QVBoxLayout(panels_widget)
        panels_layout.setContentsMargins(8, 8, 8, 8)
        panels_layout.setSpacing(8)
        
        # Add panels
        self.project_panel = ProjectOverviewPanel()
        panels_layout.addWidget(self.project_panel)
        
        self.asset_metrics_panel = AssetMetricsPanel()
        panels_layout.addWidget(self.asset_metrics_panel)
        
        self.filter_panel = FilterPanel()
        panels_layout.addWidget(self.filter_panel)
        
        panels_layout.addStretch()
        
        scroll.setWidget(panels_widget)
        splitter.addWidget(scroll)
        
        # Asset tree
        self.asset_tree = AssetTreePanel()
        splitter.addWidget(self.asset_tree)
        
        # Set splitter sizes
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter)
