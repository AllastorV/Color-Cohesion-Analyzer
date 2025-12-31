"""
Color Cohesion Analyzer - Translation System
Multi-language support with English and Turkish
"""

from typing import Dict, Callable, List

# Current language
_current_language = "en"

# Listeners for language changes
_language_change_listeners: List[Callable] = []


TRANSLATIONS = {
    "en": {
        # Window
        "app_title": "Color Cohesion Analyzer",
        "ready": "Ready",
        
        # Menu - File
        "menu_file": "&File",
        "menu_open_files": "&Open Files...",
        "menu_open_folder": "Open &Folder...",
        "menu_export": "&Export",
        "menu_export_png": "Export Palette &PNGs",
        "menu_export_json": "Export &JSON Report",
        "menu_export_ase": "Export &ASE Swatches",
        "menu_export_all": "Export &All...",
        "menu_exit": "E&xit",
        
        # Menu - View
        "menu_view": "&View",
        "menu_fit_content": "&Fit to Content",
        "menu_reset_view": "&Reset View",
        "menu_show_grid": "Show &Grid",
        
        # Menu - Analysis
        "menu_analysis": "&Analysis",
        "menu_start_analysis": "&Start Analysis",
        "menu_pause": "&Pause",
        "menu_stop": "S&top",
        
        # Menu - Help
        "menu_help": "&Help",
        "menu_about": "&About",
        
        # Toolbar
        "btn_add_files": "Add Files",
        "btn_start_scan": "▶ Start Scan",
        "btn_pause": "⏸ Pause",
        "btn_resume": "▶ Resume",
        "btn_stop": "⏹ Stop",
        "btn_export": "Export...",
        "btn_new_analysis": "🔄 New",
        "label_layout": "Layout:",
        "btn_radial": "Radial",
        "btn_grid": "Grid",
        "btn_cpu": "CPU",
        "btn_gpu": "GPU ✓",
        "btn_language": "TR",
        
        # Tooltips
        "tooltip_add_files": "Add image and video files for analysis (Ctrl+O)",
        "tooltip_start_scan": "Start color analysis of added files (Ctrl+Enter)",
        "tooltip_pause": "Pause or resume analysis",
        "tooltip_resume": "Resume paused analysis",
        "tooltip_stop": "Stop analysis completely",
        "tooltip_export": "Export analysis results as PNG, JSON, ASE or LUT",
        "tooltip_new_analysis": "Clear current analysis and start fresh (Ctrl+N)",
        "tooltip_radial": "Apply radial layout with consensus palette at center",
        "tooltip_grid": "Display all palettes in grid layout",
        "tooltip_gpu_toggle": "Toggle between CPU and GPU processing.\nGPU requires CUDA-enabled NVIDIA card.",
        "tooltip_language": "Switch language / Dil değiştir",
        "tooltip_status": "Current operation status",
        "tooltip_progress": "Analysis progress",
        "tooltip_processing_unit": "Active processing unit - CPU or GPU",
        "tooltip_file_count": "Total number of files loaded for analysis",
        
        # Drop Zone
        "drop_zone_title": "📁",
        "drop_zone_text": "Drop images and videos here\nor click to browse",
        "drop_zone_formats": "Supported: JPG, PNG, TIFF, MP4, MOV, AVI, MKV",
        
        # Dialogs
        "dialog_select_files": "Select Images and Videos",
        "dialog_select_folder": "Select Folder",
        "dialog_select_output": "Select Output Directory",
        "dialog_no_files": "No Files",
        "dialog_no_files_msg": "Please add some files first.",
        "dialog_no_results": "No Results",
        "dialog_no_results_msg": "Please run analysis first.",
        "dialog_export_complete": "Export Complete",
        "dialog_export_complete_msg": "Project exported to:\n{path}",
        "dialog_export_error": "Export Error",
        "dialog_export_error_msg": "Export failed:\n{error}",
        "dialog_analysis_error": "Analysis Error",
        "dialog_analysis_error_msg": "An error occurred:\n{error}",
        "dialog_confirm_new": "Confirm New Analysis",
        "dialog_confirm_new_msg": "This will clear the current analysis.\nAre you sure you want to continue?",
        
        # Status messages
        "status_added_files": "Added {count} files",
        "status_processing": "Processing: {filename}",
        "status_computing": "Computing consensus and metrics...",
        "status_complete": "Analysis complete",
        "status_stopped": "Analysis stopped",
        "status_paused": "Paused - Click Resume to continue",
        "status_resumed": "Resumed",
        "status_error": "Error",
        "status_cleared": "Analysis cleared - Ready for new scan",
        
        # Panels - Asset Metrics
        "panel_asset_metrics": "Asset Metrics",
        "label_palette": "Palette",
        "label_cohesion": "Cohesion",
        "label_entropy": "Entropy",
        "label_distance": "Distance",
        "label_temperature": "Temperature:",
        "label_saturation": "Saturation:",
        "label_divergent": "Divergent Colors:",
        "label_none": "None",
        "label_warm": "Warm",
        "label_cool": "Cool",
        "label_neutral": "Neutral",
        
        # Tooltips - Metrics
        "tooltip_cohesion": "Cohesion score between 0-1.\nValues close to 1 indicate high color harmony.",
        "tooltip_entropy": "Color complexity of the palette.\nHigher value = more diversity.",
        "tooltip_distance": "DeltaE distance to consensus palette.\nLower value = closer to palette.",
        "tooltip_temperature": "Color temperature (in Kelvin).\n• Warm (2700-4500K): Warm tones - candlelight, sunset\n• Neutral (4500-6500K): Natural daylight\n• Cool (6500-10000K): Cool tones - cloudy sky, blue hour",
        "tooltip_saturation": "Saturation statistics.\nMean = average saturation\nStd = saturation variance",
        "tooltip_divergent": "Colors diverging from consensus palette.\nThese differ from the project's overall color language.",
        
        # Panels - Project Overview
        "panel_project_overview": "Project Overview",
        "label_assets": "Assets:",
        "label_images": "images",
        "label_videos": "videos",
        "label_shots": "shots",
        "label_avg_cohesion": "Avg Cohesion",
        "label_outliers": "Outliers",
        "label_temp_distribution": "Temperature Distribution",
        "label_warm_bar": "Warm:",
        "label_cool_bar": "Cool:",
        "label_recommendations": "Recommendations:",
        "label_no_issues": "Analysis complete, no issues found.",
        
        # Tooltips - Project
        "tooltip_avg_cohesion": "Average cohesion score of all assets in project.\n0.8+ excellent, 0.6-0.8 good, <0.6 low",
        "tooltip_outliers": "Number of assets significantly diverging from consensus palette.\nHigh outlier rate means low visual consistency.",
        
        # Panels - Filters
        "panel_filters": "Filters",
        "label_view_mode": "View Mode",
        "view_all_assets": "All Assets",
        "view_images_only": "Images Only",
        "view_videos_only": "Videos Only",
        "view_outliers_only": "Outliers Only",
        "label_reference_mode": "Reference Mode",
        "label_display_options": "Display Options",
        "label_show_hex": "Show Hex Codes",
        "label_show_connections": "Show Connections",
        "label_compact_mode": "Compact Mode",
        "label_distance_threshold": "Distance Threshold",
        
        # Tooltips - Filters
        "tooltip_view_mode": "Select the type of assets to display",
        "tooltip_reference_mode": "Compare others against a selected reference asset",
        "tooltip_reference_select": "Select the asset to use as reference",
        "tooltip_show_hex": "Show hex codes on color swatches",
        "tooltip_show_connections": "Show connection lines between assets and palettes",
        "tooltip_compact_mode": "Reduce node sizes to show more items",
        
        # Panels - Assets
        "panel_assets": "Assets",
        "label_items": "items",
        "label_central_palettes": "Central Palettes",
        "label_consensus": "Consensus",
        "label_global_average": "Global Average",
        
        # About dialog
        "about_title": "About Color Cohesion Analyzer",
        "about_text": "Color Cohesion Analyzer v1.0\n\nA professional-grade tool for analyzing color palettes\nand evaluating visual coherence across media assets.\n\nDesigned for filmmakers and visual artists.",
        
        # Misc
        "files_count": "{count} files",
        "click_to_copy": "Click: Copy",
        "copied": "Copied!",
    },
    
    "tr": {
        # Window
        "app_title": "Renk Uyum Analizi",
        "ready": "Hazır",
        
        # Menu - File
        "menu_file": "&Dosya",
        "menu_open_files": "&Dosya Aç...",
        "menu_open_folder": "&Klasör Aç...",
        "menu_export": "&Dışa Aktar",
        "menu_export_png": "Palet &PNG'lerini Aktar",
        "menu_export_json": "&JSON Raporu Aktar",
        "menu_export_ase": "&ASE Renk Örnekleri Aktar",
        "menu_export_all": "&Tümünü Aktar...",
        "menu_exit": "&Çıkış",
        
        # Menu - View
        "menu_view": "&Görünüm",
        "menu_fit_content": "&İçeriğe Sığdır",
        "menu_reset_view": "Görünümü &Sıfırla",
        "menu_show_grid": "&Izgara Göster",
        
        # Menu - Analysis
        "menu_analysis": "&Analiz",
        "menu_start_analysis": "Analizi &Başlat",
        "menu_pause": "&Duraklat",
        "menu_stop": "&Durdur",
        
        # Menu - Help
        "menu_help": "&Yardım",
        "menu_about": "&Hakkında",
        
        # Toolbar
        "btn_add_files": "Dosya Ekle",
        "btn_start_scan": "▶ Taramayı Başlat",
        "btn_pause": "⏸ Duraklat",
        "btn_resume": "▶ Devam Et",
        "btn_stop": "⏹ Durdur",
        "btn_export": "Dışa Aktar...",
        "btn_new_analysis": "🔄 Yeni",
        "label_layout": "Düzen:",
        "btn_radial": "Dairesel",
        "btn_grid": "Izgara",
        "btn_cpu": "CPU",
        "btn_gpu": "GPU ✓",
        "btn_language": "EN",
        
        # Tooltips
        "tooltip_add_files": "Analiz için görüntü ve video dosyaları ekleyin (Ctrl+O)",
        "tooltip_start_scan": "Eklenen dosyaların renk analizini başlatın (Ctrl+Enter)",
        "tooltip_pause": "Analizi duraklatın veya devam ettirin",
        "tooltip_resume": "Duraklatılmış analizi devam ettirin",
        "tooltip_stop": "Analizi tamamen durdurun",
        "tooltip_export": "Analiz sonuçlarını PNG, JSON, ASE veya LUT olarak dışa aktarın",
        "tooltip_new_analysis": "Mevcut analizi temizle ve yeni başla (Ctrl+N)",
        "tooltip_radial": "Konsensüs paletini merkeze alarak dairesel düzen uygula",
        "tooltip_grid": "Tüm paletleri ızgara düzeninde göster",
        "tooltip_gpu_toggle": "CPU ve GPU işleme arasında geçiş yapın.\nGPU, CUDA destekli NVIDIA kartı gerektirir.",
        "tooltip_language": "Dil değiştir / Switch language",
        "tooltip_status": "Mevcut işlem durumu",
        "tooltip_progress": "Analiz ilerleme durumu",
        "tooltip_processing_unit": "Aktif işlem birimi - CPU veya GPU",
        "tooltip_file_count": "Analiz için yüklenen toplam dosya sayısı",
        
        # Drop Zone
        "drop_zone_title": "📁",
        "drop_zone_text": "Görüntü ve videoları buraya sürükleyin\nveya göz atmak için tıklayın",
        "drop_zone_formats": "Desteklenen: JPG, PNG, TIFF, MP4, MOV, AVI, MKV",
        
        # Dialogs
        "dialog_select_files": "Görüntü ve Video Seçin",
        "dialog_select_folder": "Klasör Seçin",
        "dialog_select_output": "Çıktı Dizinini Seçin",
        "dialog_no_files": "Dosya Yok",
        "dialog_no_files_msg": "Lütfen önce dosya ekleyin.",
        "dialog_no_results": "Sonuç Yok",
        "dialog_no_results_msg": "Lütfen önce analizi çalıştırın.",
        "dialog_export_complete": "Dışa Aktarım Tamamlandı",
        "dialog_export_complete_msg": "Proje şuraya aktarıldı:\n{path}",
        "dialog_export_error": "Dışa Aktarım Hatası",
        "dialog_export_error_msg": "Dışa aktarım başarısız:\n{error}",
        "dialog_analysis_error": "Analiz Hatası",
        "dialog_analysis_error_msg": "Bir hata oluştu:\n{error}",
        "dialog_confirm_new": "Yeni Analizi Onayla",
        "dialog_confirm_new_msg": "Bu işlem mevcut analizi temizleyecek.\nDevam etmek istediğinizden emin misiniz?",
        
        # Status messages
        "status_added_files": "{count} dosya eklendi",
        "status_processing": "İşleniyor: {filename}",
        "status_computing": "Konsensüs ve metrikler hesaplanıyor...",
        "status_complete": "Analiz tamamlandı",
        "status_stopped": "Analiz durduruldu",
        "status_paused": "Duraklatıldı - Devam etmek için Devam Et'e tıklayın",
        "status_resumed": "Devam edildi",
        "status_error": "Hata",
        "status_cleared": "Analiz temizlendi - Yeni tarama için hazır",
        
        # Panels - Asset Metrics
        "panel_asset_metrics": "Varlık Metrikleri",
        "label_palette": "Palet",
        "label_cohesion": "Uyum",
        "label_entropy": "Entropi",
        "label_distance": "Mesafe",
        "label_temperature": "Sıcaklık:",
        "label_saturation": "Doygunluk:",
        "label_divergent": "Sapan Renkler:",
        "label_none": "Yok",
        "label_warm": "Sıcak",
        "label_cool": "Soğuk",
        "label_neutral": "Nötr",
        
        # Tooltips - Metrics
        "tooltip_cohesion": "0-1 arası uyum skoru.\n1'e yakın değerler yüksek renk uyumu gösterir.",
        "tooltip_entropy": "Paletin renk karmaşıklığı.\nYüksek değer = daha fazla çeşitlilik.",
        "tooltip_distance": "Konsensüs paletine DeltaE mesafesi.\nDüşük değer = palete daha yakın.",
        "tooltip_temperature": "Renk sıcaklığı (Kelvin cinsinden).\n• Sıcak (2700-4500K): Sıcak tonlar - mum ışığı, gün batımı\n• Nötr (4500-6500K): Doğal gün ışığı\n• Soğuk (6500-10000K): Soğuk tonlar - bulutlu gök, mavi saat",
        "tooltip_saturation": "Doygunluk istatistikleri.\nMean = ortalama doygunluk\nStd = doygunluk varyansı",
        "tooltip_divergent": "Konsensüs paletinden sapan renkler.\nBunlar projenin genel renk dilinden farklı.",
        
        # Panels - Project Overview
        "panel_project_overview": "Proje Genel Bakış",
        "label_assets": "Varlıklar:",
        "label_images": "görüntü",
        "label_videos": "video",
        "label_shots": "çekim",
        "label_avg_cohesion": "Ort. Uyum",
        "label_outliers": "Aykırılar",
        "label_temp_distribution": "Sıcaklık Dağılımı",
        "label_warm_bar": "Sıcak:",
        "label_cool_bar": "Soğuk:",
        "label_recommendations": "Öneriler:",
        "label_no_issues": "Analiz tamamlandı, sorun bulunamadı.",
        
        # Tooltips - Project
        "tooltip_avg_cohesion": "Projedeki tüm varlıkların ortalama uyum skoru.\n0.8+ mükemmel, 0.6-0.8 iyi, 0.6- düşük",
        "tooltip_outliers": "Konsensüs paletinden önemli ölçüde sapan varlık sayısı.\nYüksek aykırı oranı düşük görsel tutarlılık demektir.",
        
        # Panels - Filters
        "panel_filters": "Filtreler",
        "label_view_mode": "Görünüm Modu",
        "view_all_assets": "Tüm Varlıklar",
        "view_images_only": "Sadece Görüntüler",
        "view_videos_only": "Sadece Videolar",
        "view_outliers_only": "Sadece Aykırılar",
        "label_reference_mode": "Referans Modu",
        "label_display_options": "Görüntüleme Seçenekleri",
        "label_show_hex": "Hex Kodlarını Göster",
        "label_show_connections": "Bağlantıları Göster",
        "label_compact_mode": "Kompakt Mod",
        "label_distance_threshold": "Mesafe Eşiği",
        
        # Tooltips - Filters
        "tooltip_view_mode": "Görüntülenecek varlık türünü seçin",
        "tooltip_reference_mode": "Seçili bir varlığı referans alarak diğerlerini karşılaştırın",
        "tooltip_reference_select": "Referans olarak kullanılacak varlığı seçin",
        "tooltip_show_hex": "Renk kutularında hex kodlarını göster",
        "tooltip_show_connections": "Varlıklar ve paletler arasındaki bağlantı çizgilerini göster",
        "tooltip_compact_mode": "Düğüm boyutlarını küçülterek daha fazla öğe görünür hale getir",
        
        # Panels - Assets
        "panel_assets": "Varlıklar",
        "label_items": "öğe",
        "label_central_palettes": "Merkezi Paletler",
        "label_consensus": "Konsensüs",
        "label_global_average": "Genel Ortalama",
        
        # About dialog
        "about_title": "Renk Uyum Analizi Hakkında",
        "about_text": "Renk Uyum Analizi v1.0\n\nRenk paletlerini analiz etmek ve medya varlıkları\narasındaki görsel tutarlılığı değerlendirmek için\nprofesyonel düzeyde bir araç.\n\nFilm yapımcıları ve görsel sanatçılar için tasarlandı.",
        
        # Misc
        "files_count": "{count} dosya",
        "click_to_copy": "Tıkla: Kopyala",
        "copied": "Kopyalandı!",
    }
}


def get_text(key: str, **kwargs) -> str:
    """Get translated text for given key"""
    text = TRANSLATIONS.get(_current_language, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    return text


def t(key: str, **kwargs) -> str:
    """Shorthand for get_text"""
    return get_text(key, **kwargs)


def get_current_language() -> str:
    """Get current language code"""
    return _current_language


def set_language(lang: str):
    """Set current language and notify listeners"""
    global _current_language
    if lang in TRANSLATIONS:
        _current_language = lang
        # Notify all listeners
        for listener in _language_change_listeners:
            try:
                listener()
            except Exception:
                pass


def toggle_language():
    """Toggle between English and Turkish"""
    global _current_language
    if _current_language == "en":
        set_language("tr")
    else:
        set_language("en")


def add_language_change_listener(callback: Callable):
    """Add a callback to be called when language changes"""
    if callback not in _language_change_listeners:
        _language_change_listeners.append(callback)


def remove_language_change_listener(callback: Callable):
    """Remove a language change listener"""
    if callback in _language_change_listeners:
        _language_change_listeners.remove(callback)
