import subprocess
import sys
import webbrowser
import threading
import pandas as pd
import numpy as np
from scipy.stats import beta, triang, norm, gamma, skewnorm
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objs as go
import io

def pip_install(pkgs):
    for p in pkgs:
        try:
            __import__(p.split('==')[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

pip_install([
    "dash", "pandas", "numpy", "plotly", "statsmodels", "scipy", 
    "dash-bootstrap-components", "dash_daq"
])

# Financial and market data configuration
data = {
    "cerebras": {
        "revenue": {'2022':24.6, '2023':78.74, '2024TTM':206.48, '2024E':272},
        "valuation": (7000, 8400, 10400),
        "growth_rate": 2.45,
        "gross_margin_2022": 0.117,
        "gross_margin_2023": 0.335,
        "gross_margin_2024": 0.411,
        "customer_concentration": 0.87,
        "employee_count": 401,
        "strategic_premium": 1.25
    },
    "intel": {
        "chips_act_funding": 7860,
        "total_investment": 100000,
        "tax_credit": 0.25,
        "rd_budget": 16520,
        "market_share_loss": (0.05, 0.15),
        "revenue": 53100,
        "ai_roadmap": {
            "Gaudi 3 AI Accelerator": "H2 2025",
            "Core Ultra Mobile (Panther Lake)": "Mid 2025",
            "Arc Graphics": "2025",
            "Foundry Services": "2025+",
            "AI Software/Tools": "2025+"
        }
    },
    "market": {
        "ai_chip_2024": 123160,
        "ai_chip_2029": 311580,
        "ai_chip_2033": 501970,
        "ai_chip_2035": 846850,
        "cagr": (0.204, 0.355),
        "current_market": 123160,
        "hbm_growth": 0.45,
        "competitive_threat": {
            "nvidia": 0.70,
            "amd": 0.15,
            "intel": 0.10,
            "others": 0.05
        }
    },
    "breakeven": {
        "upfront_cost": 5200,
        "staggered_payments": [867, 867, 867],
        "prepayment": 1430,
        "target_margin": 0.45
    }
}

def calculate_breakeven():
    years = np.arange(2025, 2036)
    revenue = [0.25, 0.45, 0.78, 1.15, 1.62, 2.31, 3.24, 4.53, 6.34, 8.88, 12.43]
    margin = [r * 0.3 for r in revenue]
    cum_cashflow = np.cumsum([-5.2] + margin).tolist()
    breakeven_year = next((i for i, v in enumerate(cum_cashflow) if v >= 0), 10)
    return years, revenue, margin, cum_cashflow, breakeven_year

def generate_analysis_csv(df, breakeven_data):
    formulas = [
        "Net Cost = Upfront Payment + Staggered Payments - Prepayment",
        "Breakeven Year = Net Cost / Average Annual Margin",
        "NPV = Σ(CF_t / (1 + r)^t) - Initial Investment",
        "Contribution Margin = Revenue * Gross Margin Percentage",
        "Strategic Premium = 1 + (Competitor Threat * Market Growth)"
    ]
    formula_df = pd.DataFrame({
        "Formula": formulas,
        "Example Calculation": [
            "5.2B + 2.6B - 1.43B = 6.37B",
            "6.37B / 1.1B ≈ 5.8 years",
            "Σ(CF_t/1.1^t) - 7.8B",
            "0.25B * 30% = 0.075B",
            "1 + (0.25 * 0.35) = 1.0875"
        ]
    })
    combined_df = pd.concat([df, formula_df], axis=0)
    return combined_df

def monte_carlo(n=5000):
    np.random.seed(42)
    res = []
    for _ in range(n):
        acq = np.random.triangular(*data['cerebras']['valuation'])
        g = np.clip(skewnorm.rvs(5, loc=2.45, scale=0.3), 0.15, 3.0)
        m = np.clip(norm.rvs(loc=data['cerebras']['gross_margin_2024'], scale=0.05), 0.2, 0.5)
        ar = np.random.uniform(0.1, 0.35)
        eff_g = g * (1 - ar)
        ms = beta.rvs(8, 2) * (1 - np.random.uniform(
            data['intel']['market_share_loss'][0],
            data['intel']['market_share_loss'][1]
        ))
        rd_inv = data['intel']['rd_budget'] * 0.2 * 5
        tax = rd_inv * data['intel']['tax_credit']
        cagr = np.random.uniform(*data['market']['cagr'])
        acq_cf = [
            data['cerebras']['revenue']['2024TTM'] * 
            (1 + eff_g)**i * m * data['cerebras']['strategic_premium']
            for i in range(1, 6)
        ]
        acq_npv = npv(acq_cf, acq, 0.10)
        mkt_5y = data['market']['current_market'] * (1 + cagr)**5
        mkt_gain = (mkt_5y - data['market']['current_market']) * ms
        rd_cf = [
            mkt_gain * 0.30 * (1 + data['market']['hbm_growth'])**i
            for i in range(5)
        ]
        rd_npv = npv(rd_cf, rd_inv - tax, 0.10)
        res.append({
            'Acq': acq_npv,
            'RD': rd_npv,
            'AttrRisk': ar,
            'MktShare': ms,
            'CAGR': cagr,
            'Breakeven': (7.8 / (acq_npv/1000)) if acq_npv > 0 else np.nan
        })
    return pd.DataFrame(res)

def npv(cfs, init, r):
    return -init + sum([cf / (1 + r)**(i + 1) for i, cf in enumerate(cfs)])

yrs = np.array([2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029])
cb_rev = np.array([24.6, 78.74, 206.48, 272, 340, 408, 480, 560])
cb_mg = np.array([11.7, 33.5, 41.1, 45, 47, 48, 49, 50])
in_rd = np.array([16520, 17000, 17500, 18000, 18500, 19000, 19500, 20000])
ai_mkt = np.array([123160, 150000, 180000, 213600, 256000, 311580, 400000, 501970])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Team Big4 Analytics Dashboard", className="text-center my-4"))),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Cerebras Valuation (2024)"),
            dbc.CardBody([
                html.H4(f"${data['cerebras']['valuation'][0]/1000}B - ${data['cerebras']['valuation'][2]/1000}B", className="card-title"),
                html.Div("Source: Sacra, 2024", className="small text-muted")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("AI Chip Market 2029"),
            dbc.CardBody([
                html.H4(f"${data['market']['ai_chip_2029']/1000:.1f}B", className="card-title"),
                html.Div("Source: MarketsandMarkets, 2024", className="small text-muted")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Cerebras Gross Margin (2024)"),
            dbc.CardBody([
                html.H4(f"{data['cerebras']['gross_margin_2024']*100:.1f}%", className="card-title"),
                html.Div("Source: Sacra, 2024", className="small text-muted")
            ])
        ]), md=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader("AI Investment 2025"),
            dbc.CardBody([
                html.H4("$200B", className="card-title"),
                html.Div("Source: Goldman Sachs, 2025", className="small text-muted")
            ])
        ]), md=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='growth-comparison'),
            dbc.Row([
                dbc.Col([
                    dcc.Checklist(
                        id='growth-toggle',
                        options=[
                            {'label': ' Cerebras Revenue', 'value': 'cerebras'},
                            {'label': ' AI Market', 'value': 'market'},
                            {'label': ' Intel R&D', 'value': 'intel'}
                        ],
                        value=['cerebras', 'market', 'intel'],
                        inline=True,
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary"
                    )
                ], md=12)
            ], className="mt-2")
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Controls"),
                dbc.CardBody([
                    html.P("Monte Carlo Simulation Count:", className="mb-1"),
                    dcc.Slider(
                        id='sim-slider',
                        min=500,
                        max=10000,
                        value=2000,
                        marks={i: f"{i}" for i in [500, 2500, 5000, 7500, 10000]},
                        step=500
                    ),
                    html.Div(id='sim-slider-output', className="mt-1 mb-3"),
                    html.P("AI Market CAGR (%):", className="mb-1"),
                    dcc.Slider(
                        id='cagr-slider',
                        min=20,
                        max=36,
                        value=28,
                        marks={i: f"{i}%" for i in range(20, 37, 4)},
                        step=1
                    ),
                    html.Div(id='cagr-slider-output', className="mt-1")
                ])
            ])
        ], md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='npv-distribution'), md=6),
        dbc.Col(dcc.Graph(id='risk-profile'), md=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='market-growth'), md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Chip Market Share 2024"),
                dbc.CardBody([
                    dcc.Graph(
                        id='market-share-pie',
                        figure=go.Figure(
                            data=[go.Pie(
                                labels=['NVIDIA', 'AMD', 'Intel', 'Others'],
                                values=[
                                    data['market']['competitive_threat']['nvidia']*100,
                                    data['market']['competitive_threat']['amd']*100,
                                    data['market']['competitive_threat']['intel']*100,
                                    data['market']['competitive_threat']['others']*100
                                ],
                                hole=.3,
                                marker_colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                                textinfo='percent+label',
                                textposition='inside'
                            )],
                            layout=dict(
                                height=300,
                                margin=dict(t=30, b=30),
                                showlegend=True,
                                annotations=[dict(
                                    text='2024',
                                    x=0.5, y=0.5,
                                    font_size=14,
                                    showarrow=False
                                )]
                            )
                        )
                    ),
                    html.Div("Source: Astute Analytica, 2025", className="small text-muted")
                ])
            ])
        ], md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Acquisition Breakeven Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='breakeven-chart'),
                    html.Div(id='breakeven-text', className="mt-2"),
                    dbc.Button("Export Analysis CSV", 
                             id='export-csv',
                             className="mt-3"),
                    dcc.Download(id="download-csv")
                ])
            ])
        ], md=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Cerebras Revenue & Gross Margin Trends"),
                dbc.CardBody([
                    dcc.Graph(
                        id='cerebras-revenue-margin',
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    x=yrs,
                                    y=cb_rev,
                                    name="Revenue (M USD)",
                                    marker_color='#636EFA',
                                    yaxis='y1'
                                ),
                                go.Scatter(
                                    x=yrs,
                                    y=cb_mg,
                                    name="Gross Margin (%)",
                                    marker_color='#EF553B',
                                    yaxis='y2',
                                    mode='lines+markers'
                                )
                            ],
                            layout=go.Layout(
                                title="Cerebras Revenue and Gross Margin (2022-2029)",
                                yaxis=dict(title="Revenue (M USD)", side='left'),
                                yaxis2=dict(title="Gross Margin (%)", overlaying='y', side='right', range=[0, 100]),
                                barmode='group',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02)
                            )
                        )
                    ),
                    html.Div("Source: Sacra, 2024", className="small text-muted")
                ])
            ])
        ], md=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Cerebras vs. GPU Inference Performance"),
                dbc.CardBody([
                    dcc.Graph(
                        id='cerebras-vs-gpu',
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    x=["Cerebras WSE-3", "Leading GPU Cloud"],
                                    y=[969, 13],
                                    marker_color=['#636EFA', '#EF553B'],
                                    text=["969 tokens/s", "13 tokens/s"],
                                    textposition='auto'
                                )
                            ],
                            layout=go.Layout(
                                title="Llama 3.1-405B Inference Speed (tokens/sec)",
                                yaxis=dict(title="Tokens/sec"),
                                xaxis=dict(title="Platform"),
                                height=350
                            )
                        )
                    ),
                    html.Div("Source: Cerebras Press Release, 2025", className="small text-muted")
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Investment Trend (Global)"),
                dbc.CardBody([
                    dcc.Graph(
                        id='ai-investment-trend',
                        figure=go.Figure(
                            data=[
                                go.Scatter(
                                    x=[2020, 2021, 2022, 2023, 2024, 2025],
                                    y=[40, 60, 90, 120, 160, 200],
                                    marker_color='#00CC96',
                                    mode='lines+markers',
                                    fill='tozeroy'
                                )
                            ],
                            layout=go.Layout(
                                title="Global AI Investment 2020-2025",
                                yaxis=dict(title="Investment (Billion USD)"),
                                xaxis=dict(title="Year"),
                                height=350
                            )
                        )
                    ),
                    html.Div("Source: Goldman Sachs, 2025", className="small text-muted")
                ])
            ])
        ], md=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Intel AI Roadmap (2025)"),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Gaudi 3 AI Accelerator: H2 2025"),
                        html.Li("Core Ultra Mobile (Panther Lake): Mid 2025"),
                        html.Li("Arc Graphics: 2025"),
                        html.Li("Foundry Services: 2025+"),
                        html.Li("AI Software/Tools: 2025+"),
                    ]),
                    html.Div("Source: AInvest, 2025", className="small text-muted")
                ])
            ])
        ], md=12)
    ], className="mb-4"),
], fluid=True)

@app.callback(
    [Output('sim-slider-output', 'children'),
     Output('cagr-slider-output', 'children')],
    [Input('sim-slider', 'value'),
     Input('cagr-slider', 'value')]
)
def show_slider(sim, cagr):
    return f"Simulations: {sim:,}", f"CAGR: {cagr}%"

@app.callback(
    Output('growth-comparison', 'figure'),
    Input('growth-toggle', 'value')
)
def plot_growth(sel):
    fig = go.Figure()
    if 'cerebras' in sel:
        fig.add_trace(go.Scatter(
            x=yrs, 
            y=cb_rev,
            mode='lines+markers',
            name='Cerebras Revenue (M USD)',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>%{x}</b><br>$%{y:.1f}M<extra></extra>'
        ))
    if 'market' in sel:
        fig.add_trace(go.Scatter(
            x=yrs,
            y=ai_mkt,
            mode='lines+markers',
            name='AI Market (M USD)',
            line=dict(color='#00CC96', width=3, dash='dot'),
            marker=dict(size=10, symbol='square'),
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>'
        ))
    if 'intel' in sel:
        fig.add_trace(go.Scatter(
            x=yrs,
            y=in_rd,
            mode='lines+markers',
            name='Intel R&D (M USD)',
            line=dict(color='#EF553B', width=3, dash='dash'),
            marker=dict(size=10, symbol='triangle-up'),
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>'
        ))
    fig.update_layout(
        title='Strategic Growth Comparison (2022-2029)',
        xaxis=dict(title='Year', tickmode='linear', dtick=1),
        yaxis=dict(
            title='Amount (Million USD)',
            type='log',
            range=[2, 6]
        ),
        hovermode='closest',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500
    )
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text="Log scale used for visibility",
        showarrow=False,
        font=dict(size=10),
        align='left'
    )
    return fig

@app.callback(
    [Output('npv-distribution', 'figure'),
     Output('risk-profile', 'figure'),
     Output('market-growth', 'figure')],
    [Input('sim-slider', 'value'),
     Input('cagr-slider', 'value')]
)
def show_analysis(n, cagr_in):
    df = monte_carlo(n)
    npv_fig = go.Figure()
    npv_fig.add_trace(go.Histogram(
        x=df['Acq'], 
        name='Acquisition', 
        marker_color='#636EFA', 
        opacity=0.7,
        nbinsx=30,
        hovertemplate='NPV: $%{x:.1f}M<br>Count: %{y}<extra></extra>'
    ))
    npv_fig.add_trace(go.Histogram(
        x=df['RD'], 
        name='R&D', 
        marker_color='#EF553B', 
        opacity=0.7,
        nbinsx=30,
        hovertemplate='NPV: $%{x:.1f}M<br>Count: %{y}<extra></extra>'
    ))
    acq_pos = (df['Acq'] > 0).mean() * 100
    rd_pos = (df['RD'] > 0).mean() * 100
    npv_fig.update_layout(
        barmode='overlay',
        title=f'NPV Distribution ({n:,} Simulations)',
        xaxis_title='Net Present Value (Million USD)',
        yaxis_title='Frequency',
        template='plotly_white',
        annotations=[
            dict(
                x=0.01, y=0.95,
                xref='paper', yref='paper',
                text=f"Acquisition Positive NPV: {acq_pos:.1f}%",
                showarrow=False,
                font=dict(color='#636EFA')
            ),
            dict(
                x=0.01, y=0.90,
                xref='paper', yref='paper',
                text=f"R&D Positive NPV: {rd_pos:.1f}%",
                showarrow=False,
                font=dict(color='#EF553B')
            )
        ],
        height=400
    )
    risk_fig = go.Figure()
    risk_fig.add_trace(go.Violin(
        y=df['Acq'],
        name='Acquisition',
        box_visible=True,
        meanline_visible=True,
        fillcolor='#636EFA',
        line_color='black',
        opacity=0.6,
        hoverinfo='none'
    ))
    risk_fig.add_trace(go.Violin(
        y=df['RD'],
        name='R&D',
        box_visible=True, 
        meanline_visible=True,
        fillcolor='#EF553B',
        line_color='black',
        opacity=0.6,
        hoverinfo='none'
    ))
    acq_m = df['Acq'].mean()
    rd_m = df['RD'].mean()
    acq_s = df['Acq'].std()
    rd_s = df['RD'].std()
    risk_fig.update_layout(
        title='Risk Profile Comparison',
        yaxis_title='Net Present Value (Million USD)',
        template='plotly_white',
        annotations=[
            dict(
                x=0.01, y=0.95,
                xref='paper', yref='paper',
                text=f"Acq Mean: ${acq_m:.1f}M, σ: ${acq_s:.1f}M",
                showarrow=False,
                font=dict(color='#636EFA', size=10)
            ),
            dict(
                x=0.01, y=0.90,
                xref='paper', yref='paper',
                text=f"R&D Mean: ${rd_m:.1f}M, σ: ${rd_s:.1f}M",
                showarrow=False,
                font=dict(color='#EF553B', size=10)
            )
        ],
        height=400
    )
    yrs_p = np.arange(2024, 2036)
    cagr = cagr_in / 100
    mkt = data['market']['current_market'] * np.power(1 + cagr, yrs_p - 2024)
    growth_fig = go.Figure()
    growth_fig.add_trace(go.Scatter(
        x=yrs_p, 
        y=mkt,
        mode='lines+markers',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>'
    ))
    growth_fig.add_vline(
        x=2029, line_width=2, line_dash="dash", line_color="#888"
    )
    growth_fig.add_annotation(
        x=2029,
        y=mkt[np.where(yrs_p == 2029)[0][0]] * 1.1,
        text="5-Year Horizon",
        showarrow=False,
        font=dict(color="#888")
    )
    growth_fig.update_layout(
        title=f'AI Chip Market Projection (CAGR: {cagr_in}%)',
        xaxis_title='Year',
        yaxis_title='Market Size (Million USD)',
        template='plotly_white',
        height=500
    )
    return npv_fig, risk_fig, growth_fig

@app.callback(
    [Output('breakeven-chart', 'figure'),
     Output('breakeven-text', 'children')],
    [Input('sim-slider', 'value')]
)
def update_breakeven(n):
    years, rev, margin, cf, beyr = calculate_breakeven()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=cf,
        mode='lines+markers',
        name='Cumulative Cash Flow',
        line=dict(color='#00CC96', width=3)
    ))
    fig.add_shape(type="line",
        x0=years[beyr], y0=0, x1=years[beyr], y1=cf[beyr],
        line=dict(color="Red",width=3,dash="dot"))
    
    fig.update_layout(
        title='Breakeven Analysis Timeline',
        xaxis_title='Year',
        yaxis_title='Cumulative Cash Flow (Billion USD)',
        template='plotly_white',
        height=400
    )
    
    text = f"Breakeven Achieved in {years[beyr]} | NPV at Breakeven: ${cf[beyr]:.2f}B"
    return fig, text

@app.callback(
    Output("download-csv", "data"),
    Input("export-csv", "n_clicks"),
    prevent_initial_call=True
)
def export_csv(n):
    if n:
        df = monte_carlo(1000)
        beyr_data = calculate_breakeven()
        combined_df = generate_analysis_csv(df, beyr_data)
        return dcc.send_data_frame(combined_df.to_csv, "acquisition_analysis.csv", index=False)
    return dash.no_update

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)