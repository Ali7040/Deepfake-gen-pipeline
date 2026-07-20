#!/usr/bin/env python3
"""
Generate PDF Documentation from Markdown
Converts the DeepTrace technical documentation to a professional PDF
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    packages = ['markdown', 'weasyprint', 'pygments']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
        except:
            print(f"Warning: Could not install {package}")

def markdown_to_html(md_file, html_file):
    """Convert Markdown to HTML with syntax highlighting"""
    import markdown
    from markdown.extensions.codehilite import CodeHiliteExtension
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.toc import TocExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Configure markdown extensions
    extensions = [
        'tables',
        'fenced_code',
        'codehilite',
        'toc',
        'nl2br',
    ]
    
    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'linenums': False,
            'guess_lang': True,
        },
        'toc': {
            'title': 'Table of Contents',
            'toc_depth': 3,
        }
    }
    
    html_content = markdown.markdown(
        md_content, 
        extensions=extensions,
        extension_configs=extension_configs
    )
    
    # Create full HTML with CSS styling
    full_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DeepTrace Technical Documentation</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        
        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #3f51b5;
            padding-bottom: 10px;
            page-break-after: avoid;
            font-size: 24pt;
            margin-top: 30px;
        }}
        
        h2 {{
            color: #303f9f;
            border-bottom: 2px solid #7986cb;
            padding-bottom: 5px;
            page-break-after: avoid;
            font-size: 18pt;
            margin-top: 25px;
        }}
        
        h3 {{
            color: #3f51b5;
            page-break-after: avoid;
            font-size: 14pt;
            margin-top: 20px;
        }}
        
        h4 {{
            color: #5c6bc0;
            font-size: 12pt;
        }}
        
        pre {{
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        
        code {{
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        
        th {{
            background-color: #3f51b5;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        tr:hover {{
            background-color: #e8eaf6;
        }}
        
        blockquote {{
            border-left: 4px solid #3f51b5;
            padding-left: 15px;
            margin-left: 0;
            color: #666;
            background-color: #f5f5f5;
            padding: 10px 15px;
        }}
        
        ul, ol {{
            margin-left: 20px;
        }}
        
        li {{
            margin-bottom: 5px;
        }}
        
        .highlight {{
            background-color: #f8f8f8;
            border-radius: 5px;
            padding: 10px;
        }}
        
        /* Python syntax highlighting */
        .highlight .k {{ color: #008000; font-weight: bold; }}  /* Keyword */
        .highlight .n {{ color: #333; }}  /* Name */
        .highlight .s {{ color: #ba2121; }}  /* String */
        .highlight .c {{ color: #408080; font-style: italic; }}  /* Comment */
        .highlight .o {{ color: #666; }}  /* Operator */
        .highlight .p {{ color: #333; }}  /* Punctuation */
        .highlight .nf {{ color: #0000ff; }}  /* Function name */
        .highlight .nb {{ color: #008000; }}  /* Builtin */
        .highlight .mi {{ color: #666; }}  /* Number */
        
        /* Cover page */
        .cover {{
            text-align: center;
            padding: 100px 0;
            page-break-after: always;
        }}
        
        .cover h1 {{
            font-size: 36pt;
            color: #1a237e;
            border: none;
            margin-bottom: 20px;
        }}
        
        .cover .subtitle {{
            font-size: 18pt;
            color: #5c6bc0;
            margin-bottom: 50px;
        }}
        
        .cover .info {{
            font-size: 14pt;
            color: #666;
            margin-top: 100px;
        }}
        
        /* Diagrams - preserve whitespace */
        .diagram {{
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre;
            font-size: 8pt;
            line-height: 1.2;
        }}
        
        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}
        
        /* Links */
        a {{
            color: #3f51b5;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* Footer */
        @page {{
            @bottom-center {{
                content: "DeepTrace Technical Documentation - Page " counter(page);
            }}
        }}
    </style>
</head>
<body>
    <div class="cover">
        <h1>🔬 DeepTrace</h1>
        <div class="subtitle">Advanced Face Swapping System</div>
        <div class="subtitle">Comprehensive Technical Documentation</div>
        <div class="info">
            <p><strong>Final Year Project</strong></p>
            <p>Deep Learning | Computer Vision | Face Processing</p>
            <p>January 2026</p>
        </div>
    </div>
    
    {html_content}
</body>
</html>'''
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return html_file


def html_to_pdf(html_file, pdf_file):
    """Convert HTML to PDF using WeasyPrint"""
    try:
        from weasyprint import HTML
        HTML(html_file).write_pdf(pdf_file)
        return True
    except Exception as e:
        print(f"WeasyPrint failed: {e}")
        return False


def generate_pdf_alternative(md_file, pdf_file):
    """Alternative: Use pandoc if available"""
    try:
        result = subprocess.run([
            'pandoc', md_file,
            '-o', pdf_file,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--toc',
            '--highlight-style=tango'
        ], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def main():
    """Main function to generate PDF documentation"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, 'DeepTrace_Technical_Documentation.md')
    html_file = os.path.join(script_dir, 'DeepTrace_Technical_Documentation.html')
    pdf_file = os.path.join(script_dir, 'DeepTrace_Technical_Documentation.pdf')
    
    if not os.path.exists(md_file):
        print(f"Error: Markdown file not found: {md_file}")
        return False
    
    print("Installing dependencies...")
    install_dependencies()
    
    print("Converting Markdown to HTML...")
    markdown_to_html(md_file, html_file)
    print(f"HTML file created: {html_file}")
    
    print("Converting HTML to PDF...")
    if html_to_pdf(html_file, pdf_file):
        print(f"✅ PDF successfully created: {pdf_file}")
        return True
    else:
        print("Trying alternative method with pandoc...")
        if generate_pdf_alternative(md_file, pdf_file):
            print(f"✅ PDF successfully created: {pdf_file}")
            return True
        else:
            print("❌ Could not generate PDF. HTML file is available.")
            print(f"   You can open {html_file} in a browser and print to PDF.")
            return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
