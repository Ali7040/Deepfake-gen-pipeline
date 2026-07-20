#!/usr/bin/env python3
"""Convert the Models In-Depth Guide to HTML"""

import markdown
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, 'DeepTrace_Models_InDepth_Guide.md')
    html_file = os.path.join(script_dir, 'DeepTrace_Models_InDepth_Guide.html')
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    css = """
        body { 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            font-size: 11pt; 
            line-height: 1.6; 
            color: #333; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        h1 { 
            color: #1a237e; 
            border-bottom: 3px solid #3f51b5; 
            padding-bottom: 10px; 
            font-size: 24pt; 
            margin-top: 30px; 
        }
        h2 { 
            color: #303f9f; 
            border-bottom: 2px solid #7986cb; 
            padding-bottom: 5px; 
            font-size: 18pt; 
            margin-top: 25px; 
        }
        h3 { 
            color: #3f51b5; 
            font-size: 14pt; 
            margin-top: 20px; 
        }
        h4 { 
            color: #5c6bc0; 
            font-size: 12pt; 
        }
        pre { 
            background-color: #f5f5f5; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 15px; 
            overflow-x: auto; 
            font-family: Consolas, Monaco, monospace; 
            font-size: 9pt; 
            line-height: 1.4; 
            white-space: pre; 
        }
        code { 
            background-color: #f0f0f0; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-family: Consolas, Monaco, monospace; 
            font-size: 10pt; 
        }
        pre code { 
            background-color: transparent; 
            padding: 0; 
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0; 
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 10px; 
            text-align: left; 
        }
        th { 
            background-color: #3f51b5; 
            color: white; 
            font-weight: bold; 
        }
        tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        .cover {
            text-align: center;
            padding: 80px 0;
            border-bottom: 2px solid #3f51b5;
            margin-bottom: 40px;
        }
        .cover h1 {
            font-size: 32pt;
            border: none;
        }
    """
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DeepTrace In-Depth Model Analysis</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="cover">
        <h1>DeepTrace Models</h1>
        <p style="font-size: 18pt; color: #5c6bc0;">In-Depth Technical Analysis</p>
        <p style="font-size: 14pt; color: #666; margin-top: 50px;">Final Year Project Documentation</p>
        <p style="color: #888;">January 2026</p>
    </div>
    
{html_content}

</body>
</html>"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML file created: {html_file}")
    print(f"Size: {os.path.getsize(html_file):,} bytes")

if __name__ == '__main__':
    main()
