from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import datetime

RISK = {
    "latrodectus": {"level": "Crítico", "desc": "Araña viuda negra — neurotóxica."},
    "loxoceles": {"level": "Crítico", "desc": "Araña violinista — veneno necrotizante."},
    "crotalus": {"level": "Alto", "desc": "Serpiente de cascabel — hemotóxica."},
    "kissing bug": {"level": "Medio", "desc": "Chinche besucona — vector Chagas."},
    "aedes": {"level": "Medio", "desc": "Mosquito Aedes — dengue, zika."},
    "lampropeltis": {"level": "Seguro", "desc": "Serpiente rey — no venenosa."},
}

def get_risk_info(label: str):
    return RISK.get(label.lower(), {"level": "Seguro", "desc": "Organismo identificado."})

def generate_pdf_report(annotated_image_bytes: bytes, detections: list) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    elements.append(Paragraph("Reporte de Detección BioRisk", title_style))
    elements.append(Spacer(1, 12))

    # Date
    date_style = styles['Normal']
    date_style.alignment = 1
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Fecha de análisis: {current_date}", date_style))
    elements.append(Spacer(1, 20))

    # Image
    img_io = BytesIO(annotated_image_bytes)
    img = Image(img_io)
    
    # Resize image to fit width (max 500)
    max_width = 450
    if img.drawWidth > max_width:
        ratio = max_width / float(img.drawWidth)
        img.drawWidth = max_width
        img.drawHeight = float(img.drawHeight) * ratio
    
    # Also restrict max height just in case it's a very tall image
    max_height = 400
    if img.drawHeight > max_height:
        ratio = max_height / float(img.drawHeight)
        img.drawHeight = max_height
        img.drawWidth = float(img.drawWidth) * ratio

    elements.append(img)
    elements.append(Spacer(1, 20))

    # Detections Table
    if detections:
        data = [["Especie / Etiqueta", "Confianza", "Riesgo", "Descripción"]]
        for det in detections:
            risk = get_risk_info(det.label)
            conf_str = f"{det.confidence * 100:.1f}%"
            data.append([det.label.capitalize(), conf_str, risk['level'], risk['desc']])
            
        table = Table(data, colWidths=[110, 70, 70, 280])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#A3E635")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No se encontraron detecciones con el umbral especificado.", styles['Normal']))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
