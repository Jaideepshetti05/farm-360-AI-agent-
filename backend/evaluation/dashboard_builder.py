import os
import json
import datetime
from backend.evaluation.config import EvalConfig

class DashboardBuilder:
    @staticmethod
    def build_dashboard(report_metadata: dict, test_cases_results: list) -> str:
        """
        Builds a beautiful, responsive, modern static HTML dashboard 
        summarizing metric scores, latency trends, and overall benchmark reports.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall_score = report_metadata.get("overall_score", 0.0) * 100
        total_cases = len(test_cases_results)
        passed_cases = sum(1 for c in test_cases_results if c.get("status") == "pass")
        failed_cases = total_cases - passed_cases
        
        # Build test cases table rows
        rows_html = ""
        for case in test_cases_results:
            status_cls = "status-pass" if case.get("status") == "pass" else "status-fail"
            rows_html += f"""
            <tr>
                <td><code>{case.get('id')}</code></td>
                <td>{case.get('category')}</td>
                <td><span class="badge {case.get('difficulty')}">{case.get('difficulty')}</span></td>
                <td>{case.get('latency', 0.0):.2f}s</td>
                <td>{case.get('score', 0.0) * 100:.1f}%</td>
                <td><span class="status-indicator {status_cls}">{case.get('status').upper()}</span></td>
            </tr>
            """
            
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Farm360 AI - Evaluation Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f6f8;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .card-number {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2a5298;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f7f9fa;
            font-weight: 600;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            text-transform: uppercase;
        }}
        .badge.easy {{ background: #e2f0d9; color: #385723; }}
        .badge.medium {{ background: #fff2cc; color: #7f6000; }}
        .badge.hard {{ background: #fce4d6; color: #c65911; }}
        
        .status-indicator {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        .status-pass {{ background-color: #d4edda; color: #155724; }}
        .status-fail {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Farm360 AI - Quality Assurance & Performance Dashboard</h1>
            <p>Generated at: {now_str} | Target Score Gate: {report_metadata.get('target_score', 0.85) * 100}%</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Overall Quality Score</h3>
                <div class="card-number">{overall_score:.1f}%</div>
            </div>
            <div class="card">
                <h3>Total Test Cases</h3>
                <div class="card-number">{total_cases}</div>
            </div>
            <div class="card">
                <h3>Passed Cases</h3>
                <div class="card-number" style="color: #28a745;">{passed_cases}</div>
            </div>
            <div class="card">
                <h3>Failed Cases</h3>
                <div class="card-number" style="color: #dc3545;">{failed_cases}</div>
            </div>
        </div>

        <h2>Test Case Details</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Difficulty</th>
                    <th>Latency</th>
                    <th>Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        os.makedirs(EvalConfig.DASHBOARDS_DIR, exist_ok=True)
        dest_path = os.path.join(EvalConfig.DASHBOARDS_DIR, "index.html")
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return dest_path
