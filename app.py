# Framework principal pour créer l'application web
import dash 

# Modules standards pour la gestion des flux de données (comme les fichiers) et pour l'encodage/décodage en base64 (utile pour les fichiers uploadés dans Dash)
import io       # Utilisé pour manipuler des fichiers en mémoire
import base64   # Permet d'encoder et de décoder des fichiers en base64, format utilisé dans Dash pour transmettre les fichiers via le web

# Modules pour interagir avec le système de fichiers et effectuer des recherches ou validations via des expressions régulières
import os       # Permet d'interagir avec le système de fichiers (par ex : vérifier si un fichier existe, créer un dossier temporaire, etc.)
import re       # Permet de manipuler des expressions régulières, utile pour valider des noms de fichiers, des URL, des types de variables, etc.

# Gestion des dates/heures pour les rapports et exports
import datetime

# Composants de base de Dash (Contrôles, HTML, callbacks)
from dash import dcc, html, Input, Output, State, dash_table, callback_context

# Preprocessing des données : Normalisation/Standardisation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# Import KNN imputer for missing values
from sklearn.impute import KNNImputer

# Exception pour arrêter la mise à jour des callbacks
from dash.exceptions import PreventUpdate

# Thèmes Bootstrap pour l'UI
import dash_bootstrap_components as dbc

# Manipulation de données tabulaires
import pandas as pd

# Calculs numériques et manipulations de tableaux
import numpy as np

# Sélection dynamique de composants dans les callbacks
from dash import ALL

# Création de figures complexes Plotly
import plotly.figure_factory as ff

# Création de sous-graphiques combinés
from plotly.subplots import make_subplots

# Génération manuelle de graphiques Plotly
import plotly.graph_objects as go

# Syntaxe simplifiée pour les graphiques Plotly (wrapper haut niveau)
import plotly.express as px

# Importation des tests statistiques de scipy.stats
from scipy.stats import shapiro, pearsonr, chi2_contingency, ttest_ind, spearmanr, kstest

# Importation de Dash Bootstrap Components pour améliorer l'esthétique du dashboard
import dash_bootstrap_components as dbc

# Importation de PreventUpdate, une exception utilisée pour empêcher le déclenchement d'un callback dans certaines conditions
from dash.exceptions import PreventUpdate

# Importation de plotly.io pour l'exportation des graphiques (ex. en PNG, JPEG, etc.)
import plotly.io as pio  # Pour l'export des graphiques

# Importation de no_update de Dash pour éviter des mises à jour dans un callback (utile pour ne pas modifier l'état d'un composant)
from dash import no_update  # Déjà présent mais important pour la gestion des callbacks

# Importation de stats de scipy pour avoir accès à diverses fonctions statistiques
from scipy import stats

#-------------------
# Initialize the app
#-------------------

app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               suppress_callback_exceptions=True,
               title="Data Analysis Dashboard",
               meta_tags=[{'name': 'viewport', 
                          'content': 'width=device-width, initial-scale=1.0'}])
global_df = None

# Store components 
stores = html.Div([
    dcc.Store(id='store-data', storage_type='memory'),
    dcc.Store(id='conversion-data-store', storage_type='memory', data=[]),
    dcc.Store(id='export-data', storage_type='memory'),
])

# Hidden components that need to exist in the layout
hidden_components = html.Div([
    html.Div(id='replace-validation', style={'display': 'none'}),
    html.Div(id='replace-preview', style={'display': 'none'}),
    html.Div(id='replace-confirm-button-container', style={'display': 'none'}),
    
    # Components that must exist at application start
    dcc.Checklist(id='replace-mean', value=[], style={'display': 'none'}),
    dcc.Checklist(id='replace-knn', value=[], style={'display': 'none'}),
    dcc.Checklist(id='replace-zero', value=[], style={'display': 'none'}),
    dcc.Checklist(id='replace-mode', value=[], style={'display': 'none'}),  # Added replace-mode component
    
    # Define knn-n-neighbors as a number input to match the dynamic creation
    dcc.Input(id='knn-n-neighbors', type='number', value=5, min=1, max=20, step=1, style={'display': 'none'}),
    
    # Define knn-aggregation as a dropdown to match the dynamic creation
    dcc.Dropdown(id='knn-aggregation', options=[
        {'label': 'Moyenne', 'value': 'mean'},
        {'label': 'Médiane', 'value': 'median'}
    ], value='mean', style={'display': 'none'}),
    
    # Add type-conversion-table as a hidden component
    html.Div([
        dash_table.DataTable(
            id='type-conversion-table',
            data=[],
            columns=[
                {'name': 'variable', 'id': 'variable'},
                {'name': 'current_type', 'id': 'current_type'},
                {'name': 'new_type', 'id': 'new_type'}
            ]
        )
    ], style={'display': 'none'}),
], style={'display': 'none'})

#___________________
#Structure de Layout
#___________________

#---------
# Sidebar
#---------
sidebar = dbc.Container([
    html.Div(
        html.H3("ExploraDonnées", className="text-white text-center fw-bold mb-4"),
        style={'borderBottom': '1px solid rgba(255,255,255,0.2)', 'paddingBottom': '15px'}
    ),
    dbc.Nav([
        dbc.NavLink("Téléchargement et Résumé des données", href="/upload", active="exact", className="text-white"),
        dbc.NavLink("Prétraitement des données", href="/preprocessing", active="exact", className="text-white"),
        
        # Bouton de téléchargement principal
        html.Div(
            dbc.Button(
                "💾 Exporter les données finales",
                id='btn-main-download',
                color="success",
                className="w-100 mt-3",
                disabled=True
            ),
            className="p-2"
        ),
        
        dbc.NavLink("Analyse Exploratoire", href="/visualization", active="exact", className="text-white"),
        dbc.NavLink("Tests Statistiques", href="/tests", active="exact", className="text-white"),
        dbc.NavLink("Bloc-notes", href="/notebook", active="exact", className="text-white"),
        
        # Espacer pour pousser les boutons vers le bas
        html.Div(style={'flexGrow': '1'}),
        
        # Conteneur pour les boutons parfaitement alignés
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',  # Deux colonnes de largeur égale
                'gap': '10px',  # Espacement entre boutons
                'width': '100%',
                'padding': '0 10px',
                'marginBottom': '15px'
            },
            children=[
                dbc.Button(
                    [html.I(className="fas fa-home me-2"), "Accueil"],
                    href="/",
                    color="primary",
                    className="w-100",
                    style={
                        'borderRadius': '5px',
                        'padding': '8px 0',
                        'width': '100%'  # Prend toute la largeur disponible
                    }
                ),
                dbc.Button(
                    [html.I(className="fas fa-question-circle me-2"), "Aide"],
                    href="/help",
                    color="primary",
                    className="w-100",
                    style={
                        'borderRadius': '5px',
                        'padding': '8px 0',
                        'width': '100%'  # Prend toute la largeur disponible
                    }
                )
            ]
        )
    ], 
    vertical=True, 
    pills=True, 
    className="flex-column h-100 d-flex",
    style={'flex': '1'}
    ),
], 
style={
    "position": "fixed",
    "top": "0",
    "left": "0",
    "height": "100vh",
    "width": "250px",
    "background-color": "#1E3A8A",
    "padding": "1rem",
    "display": "flex",
    "flexDirection": "column",
    "zIndex": 1000
})

# Le reste du code (display_area, app.layout, etc.) reste inchangé
# Modifiez le display_area pour commencer dès le début de la page
display_area = html.Div(
    id='page-content', 
    className="p-4", 
    style={
        "margin-left": "250px",
        "background-color": "#E5E7EB", 
        "min-height": "100vh",
        "padding-top": "0"  # Supprime l'espace en haut
    }
)

# Modal de configuration du téléchargement
download_modal = dbc.Modal([
    dbc.ModalHeader(html.H4("Exporter les données finales", className="fw-bold")),
    dbc.ModalBody([
        dbc.Alert(id='download-alert', is_open=False, duration=4000),
        dcc.Input(
            id='export-filename',
            type='text',
            placeholder='Nom du fichier (sans extension)',
            className='form-control-lg',
            maxLength=40
        ),
        dbc.FormText(
            "Caractères autorisés: lettres, chiffres, tirets et underscores",
            className="text-muted mt-2"
        )
    ]),
    dbc.ModalFooter([
        dbc.Button("Annuler", id="cancel-export", outline=True, className="me-2"),
        dbc.Button("Exporter", id="confirm-export", color="success")
    ])
], id="export-modal", size="lg")


#------------------
# Contenu principal
#------------------

display_area = html.Div(id='page-content', className="p-4", style={"margin-left": "260px", "background-color": "#E5E7EB", "min-height": "100vh"})

#-----------------
# Layout principal
#-----------------

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    display_area,
    # Include the store components
    stores,
    # Include hidden components
    hidden_components,
    dcc.Download(id='final-export-trigger'),
    download_modal,
    
    # Modal de profil - version corrigée
    dbc.Modal(
        [
            dbc.ModalHeader("Mon Profil Professionnel"),
            dbc.ModalBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H4("Lamouchi Sarra", className="mb-3 text-primary"),
                                    html.P(
                                        [
                                            html.I(className="fas fa-graduation-cap me-2"),
                                            "Élève-Ingénieur en Data Science"
                                        ],
                                        className="lead"
                                    ),
                                    html.P(
                                        [
                                            html.I(className="fas fa-university me-2"),
                                            "Ecole Supérieure de la Statistique et de l'Analyse de l'Information"
                                        ]
                                    ),
                                    html.Hr(),
                                    html.P(
                                        [
                                            html.I(className="fas fa-envelope me-2"),
                                            html.A(
                                                "lamouchisara34@gmail.com",
                                                href="lamouchisara34@gmail.com",
                                                className="text-decoration-none"
                                            )
                                        ]
                                    ),
                                    html.P(
                                        [
                                            html.I(className="fab fa-linkedin me-2"),
                                            html.A(
                                                "LinkedIn",
                                                href=" www.linkedin.com/in/Sarra-Lamouchi",
                                                target="_blank",
                                                className="text-decoration-none"
                                            )
                                        ]
                                    ),
                                    html.Hr(),
                                    html.P(
                                        "Créateur de cette application d'analyse de données avec Dash",
                                        className="text-muted"
                                    )
                                ]
                            )
                        ],
                        className="align-items-center"
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Fermer",
                    id="close-profile",
                    className="ms-auto",
                    color="primary"
                )
            )
        ],
        id="profile-modal",
        size="md",
        is_open=False
    )
])

@app.callback(
    Output("profile-modal", "is_open"),
    [Input("btn-profile", "n_clicks"),
     Input("close-profile", "n_clicks")],
    [State("profile-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_profile_modal(open_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id in ["btn-profile", "close-profile"]:
        return not is_open
    return is_open

def format_numeric_values(df):
    """Format all float columns to 2 decimal places and return formatted DataFrame"""
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    float_cols = df.select_dtypes(include=['float64', 'float32', 'float']).columns
    for col in float_cols:
        # Round only if there are decimal values
        if (df[col] % 1 != 0).any():
            df[col] = df[col].round(2)
    return df

#---------------------------------------------------------------------------------------------------------------------------
#téléchargement aprés modifications (partie prétraitement des données) bouton télechargement des données aprés modifictaion 
#---------------------------------------------------------------------------------------------------------------------------

# Callback pour gérer le workflow complet
@app.callback(
    [Output('export-modal', 'is_open'),
     Output('final-export-trigger', 'data'),
     Output('download-alert', 'children'),
     Output('download-alert', 'color'),
     Output('download-alert', 'is_open')],
    [Input('btn-main-download', 'n_clicks'),
     Input('confirm-export', 'n_clicks'),
     Input('cancel-export', 'n_clicks')],
    [State('store-data', 'data'),
     State('export-filename', 'value'),
     State('export-modal', 'is_open')]
)

def handle_export_workflow(btn_clicks, confirm_clicks, cancel_clicks, data, filename, is_open):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'btn-main-download':
        return True, None, None, None, False
    
    if trigger_id == 'cancel-export':
        return False, None, None, None, False

    if trigger_id == 'confirm-export':
        if not data or not filename:
            return dash.no_update, None, "Données ou nom de fichier manquant", "danger", True
            
        try:
            if not re.match(r'^[\w-]{3,40}$', filename):
                raise ValueError("Nom de fichier invalide")

            df = pd.DataFrame(data)
            df = format_numeric_values(df)  # Format before export
            
            export_dir = os.path.join('exports', datetime.datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(export_dir, exist_ok=True)
            
            sanitized_name = re.sub(r'[^\w-]', '', filename)
            full_path = os.path.join(export_dir, f"{sanitized_name}.csv")
            
            if os.path.exists(full_path):
                raise FileExistsError("Un fichier avec ce nom existe déjà")

            df.to_csv(full_path, index=False, encoding='utf-8-sig', float_format="%.2f")
            
            return (
                False,
                dcc.send_file(full_path),
                f"Export réussi : {full_path}",
                "success",
                True
            )

        except Exception as e:
            return (
                dash.no_update,
                None,
                f"Erreur d'export : {str(e)}",
                "danger",
                True
            )

    return dash.no_update, None, None, None, False


# Activation conditionnelle du bouton principal
@app.callback(
    Output('btn-main-download', 'disabled'),
    [Input('store-data', 'modified_timestamp')],
    [State('store-data', 'data')]
)
def update_download_button(ts, data):
    return not data or len(data) == 0

#-----------------------
# Mise à jour de la page
#-----------------------

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('store-data', 'data')  # Ajout de State pour récupérer les données stockées
)
def update_page(pathname, stored_data):  # Acceptation de 2 arguments
    global global_df
    df = pd.DataFrame(stored_data)
    # Page d'accueil
    if pathname == '/' or pathname is None:
     return html.Div([
        # Conteneur principal
        dbc.Container(
            fluid=True,
            style={
                'background': 'linear-gradient(135deg, #1a6fc9 0%, #0d47a1 100%)',
                'padding': '3rem 0',
                'position': 'relative',
                'minHeight': '60vh',  # Hauteur réduite
                'marginBottom': '2rem'
            },
            children=[
                # Icône profil avec texte "Profil Fondateur"
                html.Div(
                    [
                        html.I(
                            className="fas fa-user-circle",
                            id="btn-profile",
                            style={
                                "color": "white",
                                "cursor": "pointer",
                                "fontSize": "2rem",
                                "display": "block",
                                "margin": "0 auto 5px",
                                "textAlign": "center"  # Centrage horizontal
                            }
                        ),
                        html.Small(
                            "Profil Fondateur",
                            style={
                                "color": "white",
                                "display": "block",
                                "textAlign": "center",
                                "fontSize": "0.8rem",
                                "width": "100%"  # Prend toute la largeur
                            }
                        )
                    ],
                    style={
                        "position": "absolute",
                        "top": "20px",
                        "right": "20px",
                        "zIndex": 1000
                    }
                ),
                
                # Contenu centré
                dbc.Row(
                    dbc.Col(
                        [
                            html.H1(
                                "Nous sommes ravis de vous accueillir dans cet environnement conçu pour explorer, analyser et visualiser vos données en toute simplicité",
                                className="text-white text-center mb-4",
                                style={
                                    'fontSize': '2rem',
                                    'maxWidth': '800px',
                                    'margin': '0 auto',
                                    'paddingTop': '2rem',
                                    'lineHeight': '1.4'
                                    
                                }
                            ),
                            html.P(
                                "Découvrez des insights précieux à partir de vos données grâce à nos outils puissants et intuitifs.",
                                className="text-white-50 text-center mb-5",
                                style={
                                    'fontSize': '1.25rem',
                                    'maxWidth': '700px',
                                    'margin': '0 auto'
                                }
                            ),
                            html.Div(
                                dbc.Button(
                                    "Commencer →",
                                    href="/upload",
                                    color="light",
                                    size="lg",
                                    className="px-5",
                                    style={
                                        'borderRadius': '50px',
                                        'fontWeight': '400',
                                        'margin': '0 auto',
                                        'display': 'block'
                                    }
                                ),
                                className="text-center"
                            )
                        ],
                        width=12,
                        className="d-flex flex-column align-items-center",
                        style={
                            'justifyContent': 'center',
                            'padding': '2rem 0'
                        }
                    )
                )
            ]
        ),
        
        # Section guide
        dbc.Container(
            [
                html.Div(
                    [
                        html.P(
                            "Explorez le plein potentiel de notre plateforme avec nos ressources complètes",
                            className="text-center mb-3",
                            style={
                                'fontSize': '1.1rem',
                                'color': '#495057',
                                'maxWidth': '600px',
                                'margin': '0 auto'
                            }
                        ),
                        html.Div(
                            dbc.Button(
                                "Accéder aux guides",
                                href="/help",
                                color="primary",
                                className="fw-bold",
                                style={
                                    'padding': '8px 25px',
                                    'borderRadius': '50px'
                                }
                            ),
                            className="text-center"
                        )
                    ],
                    className="py-5",
                    style={
                        'borderTop': '1px solid #eee',
                        'borderBottom': '1px solid #eee',
                        'margin': '0 auto',
                        'maxWidth': '800px'
                    }
                )
            ],
            fluid=True,
            style={
                'backgroundColor': '#E5E7EB',
                'padding': '2rem 0'
            }
        )
    ],
    style={
        'backgroundColor': '#E5E7EB'
    })

    #-------------------------------------------
# Partie aide
#-------------------------------------------

    elif pathname == '/help':
     return html.Div([
        # En-tête avec dégradé de bleu
        html.Div(
            [
                html.H1("Centre d'Aide", className="display-4 text-white mb-3"),
                html.P("Ressources et support pour votre analyse de données", 
                      className="lead text-white-50 mb-4"),
            ],
            className="py-5 px-4 mb-4 text-center",
            style={
                'background': 'linear-gradient(135deg, #1a6fc9 0%, #0d47a1 100%)',
                'borderRadius': '0.5rem',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'marginTop': '-2rem',  # Compense le padding du conteneur parent
                'marginLeft': '-1rem',
                'marginRight': '-1rem'
            }
        ),
        
        # Contenu principal avec fond blanc et ombre
        dbc.Card(
            dbc.CardBody([
                # Section 1 - Résumé Statistique
                html.H2("1. Résumé Statistique", className="mt-5 text-primary border-bottom pb-2"),
                
                html.H3("Importation des Données", className="mt-3 text-dark"),
                html.Ul([
                    html.Li([
                        html.Strong("Formats supportés: ", className="text-dark"), 
                        html.Span("CSV (.csv), Excel (.xls, .xlsx), Texte (.txt)", className="text-secondary")
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Taille maximale: ", className="text-dark"), 
                        html.Span("Jusqu'à 100 Mo (selon la mémoire disponible)", className="text-secondary")
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Options: ", className="text-dark"), 
                        html.Span("Réinitialisation possible des données importées", className="text-secondary")
                    ], className="mb-2")
                ], className="mb-4"),
                
                html.H3("Variables Quantitatives", className="mt-4 text-dark"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Statistique", className="text-dark bg-light"), 
                        html.Th("Description", className="text-dark bg-light"),
                        html.Th("Interprétation", className="text-dark bg-light")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td("count", className="text-secondary"), 
                            html.Td("Nombre de valeurs non-nulles", className="text-secondary"),
                            html.Td("Identifie les valeurs manquantes", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("mean", className="text-secondary"), 
                            html.Td("Moyenne arithmétique", className="text-secondary"),
                            html.Td("Tendance centrale", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("std", className="text-secondary"), 
                            html.Td("Écart-type", className="text-secondary"),
                            html.Td("Dispersion des données", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("min/max", className="text-secondary"), 
                            html.Td("Valeurs extrêmes", className="text-secondary"),
                            html.Td("Plage des valeurs", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("25%/50%/75%", className="text-secondary"), 
                            html.Td("Quartiles", className="text-secondary"),
                            html.Td("Distribution des données", className="text-secondary")
                        ])
                    ])
                ], bordered=True, hover=True, className="mb-4"),
                
                html.H3("Variables Qualitatives", className="mt-4 text-dark"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Statistique", className="text-dark bg-light"), 
                        html.Th("Description", className="text-dark bg-light")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td("count", className="text-secondary"), 
                            html.Td("Nombre d'observations non-nulles", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("unique", className="text-secondary"), 
                            html.Td("Nombre de catégories distinctes", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("top", className="text-secondary"), 
                            html.Td("Catégorie la plus fréquente", className="text-secondary")
                        ]),
                        html.Tr([
                            html.Td("freq", className="text-secondary"), 
                            html.Td("Fréquence de la catégorie top", className="text-secondary")
                        ])
                    ])
                ], bordered=True, hover=True, className="mb-4"),
                
                html.H3("Fonctionnalités Avancées", className="mt-4 text-dark"),
                html.Ul([
                    html.Li([
                        html.Strong("Réinitialisation: ", className="text-dark"),
                        html.Span("Bouton 'Réinitialiser les données' pour supprimer le jeu de données actuel", className="text-secondary")
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Recherche: ", className="text-dark"),
                        html.Span("Filtrage des variables par nom pour les grands jeux de données", className="text-secondary")
                    ], className="mb-2")
                ], className="mb-4"),
                
                dbc.Alert([
                    html.I(className="fas fa-lightbulb me-2 text-warning"),
                    html.Span("Conseil: Consultez toujours ce résumé avant de commencer votre analyse pour détecter d'éventuels problèmes.", className="text-dark")
                ], color="light", className="mt-3 border border-warning"),
                
                # Section 2 - Prétraitement
                html.H2("2. Prétraitement des Données", className="mt-5 text-primary border-bottom pb-2"),
                
                html.H3("Gestion des Données", className="mt-3 text-dark"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Valeurs Manquantes", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Imputation par moyenne", className="text-secondary"),
                            html.Li("Méthode KNN (k-plus proches voisins)", className="text-secondary"),
                            html.Li("Imputation par Zéro", className="text-secondary")
                        ])
                    ], width=4),
                    
                    dbc.Col([
                        html.H4("Normalisation", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("StandardScaler (moyenne=0, σ=1)", className="text-secondary"),
                            html.Li("MinMaxScaler (valeurs entre 0-1)", className="text-secondary"),
                            html.Li("RobustScaler (résistant aux outliers)", className="text-secondary"),
                            html.Li("Logarithmique", className="text-secondary")
                        ])
                    ], width=4),
                    
                    dbc.Col([
                        html.H4("Suppression des doublons", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Soit Garder la première occurrence", className="text-secondary"),
                            html.Li("Soit Supprimer tous les doublons", className="text-secondary")
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                    html.Span("Les variables Normalisées sont suffixées par '_norm'", className="text-dark")
                ], color="light", className="mt-3 border border-warning"),
                
                # Section 3 - Visualisation
                html.H2("3. Visualisation des Données", className="mt-5 text-primary border-bottom pb-2"),
                
                dbc.Row([
                    dbc.Col([
                        html.H3("Visualisations Univariées", className="mt-3 text-dark"),
                        html.H4("Quantitatives", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Histogrammes", className="text-secondary"),
                            html.Li("Boxplots", className="text-secondary"),
                            html.Li("Graphiques de densité", className="text-secondary")
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Qualitatives", className="fw-bold text-dark mt-3"),
                        html.Ul([
                            html.Li("Diagrammes en barres", className="text-secondary"),
                            html.Li("Camemberts", className="text-secondary"),
                            html.Li("Diagrammes en treillis", className="text-secondary")
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                html.H3("Analyse Mixte", className="mt-4 text-dark"),
                html.Ul([
                    html.Li("Boxplots groupés (comparaisons entre catégories)", className="text-secondary"),
                    html.Li("Strip plots (distribution + individus)", className="text-secondary"),
                    html.Li("Barplots (moyennes par groupe)", className="text-secondary")
                ], className="mb-4"),
                
                html.H3("Analyses Multivariées", className="mt-4 text-dark"),
                html.Ul([
                    html.Li("Matrice de corrélation", className="text-secondary"),
                    html.Li("Scatter plots matriciels", className="text-secondary"),
                    html.Li("Heatmaps", className="text-secondary")
                ], className="mb-4"),
                
                html.H3("Analyse de Distribution", className="mt-4 text-dark"),
                html.Ul([
                    html.Li("Densité (estimation non-paramétrique)", className="text-secondary"),
                    html.Li("QQ-plots (normalité)", className="text-secondary")
                ], className="mb-4"),
                
                # Section 4 - Tests Statistiques
                html.H2("4. Tests Statistiques", className="mt-5 text-primary border-bottom pb-2"),
                
                html.H3("Tests de Normalité", className="mt-3 text-dark"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Shapiro-Wilk", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Recommandé pour n < 50", className="text-secondary"),
                            html.Li("Test précis pour petits échantillons", className="text-secondary"),
                            html.Li("Hypothèse: normalité", className="text-secondary")
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Kolmogorov-Smirnov", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Pour n ≥ 50", className="text-secondary"),
                            html.Li("Comparaison à distribution normale", className="text-secondary"),
                            html.Li("Moins puissant que Shapiro", className="text-secondary")
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                html.H3("Tests de Corrélation", className="mt-4 text-dark"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Pearson", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Corrélation linéaire", className="text-secondary"),
                            html.Li("Intervalle [-1,1]", className="text-secondary"),
                            html.Li("Exige normalité", className="text-secondary")
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Spearman", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Corrélation monotone", className="text-secondary"),
                            html.Li("Basé sur les rangs", className="text-secondary"),
                            html.Li("Non paramétrique", className="text-secondary")
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                html.H3("Test d'Indépendance Chi²", className="mt-4 text-dark"),
                html.Ul([
                    html.Li("Variables catégorielles", className="text-secondary"),
                    html.Li("Effectifs ≥ 5 par case", className="text-secondary"),
                    html.Li("Test d'association", className="text-secondary")
                ], className="mb-4"),
                
                html.H3("Test t de Student", className="mt-4 text-dark"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Variantes", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Indépendant (2 groupes)", className="text-secondary")
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Conditions", className="fw-bold text-dark"),
                        html.Ul([
                            html.Li("Normalité", className="text-secondary")
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Alert([
                    html.I(className="fas fa-lightbulb me-2 text-info"),
                    html.Span("Conseil: Vérifiez toujours les conditions d'application avant d'interpréter les résultats.", className="text-dark")
                ], color="light", className="mt-3 border border-info"),
                
                # Section 5 - Bloc-notes
                html.H2("5. Bloc-notes Analytique", className="mt-5 text-primary border-bottom pb-2"),
                html.H3("Fonctionnalités", className="mt-3 text-dark"),
                html.Ul([
                    html.Li("Sauvegarde automatique", className="text-secondary"),
                    html.Li("Export au format texte (.txt)", className="text-secondary")
                ], className="mb-4"),
                
                ], className="bg-white", style={'borderRadius': '0.5rem'}),
            className="shadow-sm border-0"
        )
    ], className="p-4", style={
        'backgroundColor': '#E5E7EB',  # Gris clair comme fond principal
        'minHeight': '100vh'
    })

    #----------------------------------
    # Page de téléchargement des données
    #----------------------------------

    elif pathname == '/upload':
        return html.Div([
            html.H4("Zone de Téléchargement des Fichiers:", style={
        'color': '#1E3A8A',  # Bleu foncé
        'marginBottom': '20px',
        'fontWeight': 'bold'
    } # Même bleu que le titre principal
                  ),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt", style={"fontSize": "24px", "color": "#007bff"}),
                    html.Br(),
                    html.Span('Déposez un fichier ici ou '),
                    html.A('cliquez pour sélectionner un fichier', className="text-primary fw-bold")
                ]),
                style={'width': '100%', 'height': '80px', 'borderWidth': '2px', 'borderStyle': 'solid',
                       'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px', 'padding': '10px',
                       'backgroundColor': '#f8f9fa', 'cursor': 'pointer', 'borderColor': '#007bff'},
                multiple=False
            ),
            html.Div(id='output-message', className='mt-2 text-success fw-bold'),
            html.Div([
                 dbc.Button("Réinitialiser les données", id="reset-btn", color="danger", className="mb-3"),
                 html.Div(id="reset-message", className="text-success fw-bold mt-2")
            ], style={'textAlign': 'right' }), 
            # Champ de recherche dynamique ajouté ici
            html.Div(id='output-data-table', className='mt-4'),

            html.H4(
               "Recherche de Variables:",
               style={
                  
                  'color': '#1E3A8A',  # Même bleu que le titre principal
                  'marginTop': '30px',
                  'marginBottom': '15px',
                  'fontWeight': 'bold'
                  }
            ),
            dcc.Input(
                id="search-input", 
                type="text", 
                placeholder="Recherchez une variable...",
                debounce=True,  # Ajoute un léger délai pour améliorer la performance
                className="mb-3",
                style={"width": "100%", "padding": "10px", "borderRadius": "5px", "border": "1px solid #ccc"}
            ),
            html.Div(id="filtered-table", className="mt-3"),

            html.Br(),
        # Bouton Résumé modifié ici
            dbc.Button(
             "Résumé", 
             id="summary-btn", 
             href="/summary", 
             color="primary",
             className="mt-3 w-100",
             style={"width": "100%"}
        )
        ])
      
    #----------------------------------
    # Page de résumé des données:
    #----------------------------------

    if pathname == '/summary':
        if stored_data is None or not stored_data:
            return html.Div(
            "⚠️ Aucune donnée disponible. Veuillez télécharger un fichier.",

            style={
                'fontSize': '24px',  # Taille augmentée
                'color': '#721c24',  # Texte rouge foncé
                'backgroundColor': '#f8d7da',  # Fond rouge clair
                'border': '1px solid #f5c6cb',  # Bordure rouge
                'padding': '20px',
                'borderRadius': '5px',
                'margin': '40px auto',
                'maxWidth': '800px',
                'fontWeight': 'bold'
            }
        ),

        global_df = pd.DataFrame(stored_data)
        quantitative_df = global_df.select_dtypes(include=['number'])
        qualitative_df = global_df.select_dtypes(exclude=['number'])

        if not quantitative_df.empty:
            summary_quantitative = quantitative_df.describe().transpose()
            summary_quantitative = format_numeric_values(summary_quantitative)
            summary_quantitative['Valeurs manquantes'] = quantitative_df.isnull().sum()
        else:
            summary_quantitative = pd.DataFrame(columns=["Aucune variable quantitative trouvée"])

        if not qualitative_df.empty and not qualitative_df.columns.empty:
            summary_qualitative = qualitative_df.describe(include=['object']).transpose()
            summary_qualitative['Valeurs manquantes'] = qualitative_df.isnull().sum()
        else:
            summary_qualitative = pd.DataFrame(columns=["Aucune variable qualitative trouvée"])

        return html.Div([
            html.H3(
            "Résumé des Données",
            style={
                'textAlign': 'center',
                'color': '#1E3A8A',
                'marginBottom': '20px',
                'fontWeight': 'bold'
            }),
            dbc.Button("← Retour", href="/upload", color="secondary", className="mb-3"),
            html.H4("Résumé des variables quantitatives", className="text-primary"),
            dash_table.DataTable(
                data=summary_quantitative.reset_index().to_dict('records'),
                columns=[{
                    'name': col,
                    'id': col,
                    'type': 'numeric',
                    'format': dash_table.Format.Format(
                        precision=2,
                        scheme=dash_table.Format.Scheme.fixed
                    ) if col != 'index' and pd.api.types.is_numeric_dtype(summary_quantitative[col]) else {}
                } for col in summary_quantitative.reset_index().columns],
                style_table={'overflowX': 'auto'},
                page_size=10
            ),
            html.H4("Résumé des variables qualitatives", className="mt-4 text-primary"),
            dash_table.DataTable(
                data=summary_qualitative.reset_index().to_dict('records'),
                columns=[{'name': col, 'id': col} for col in summary_qualitative.reset_index().columns],
                style_table={'overflowX': 'auto'},
                page_size=10
            ),
        ])
    
    #-----------------------------------
    # Page de prétraitement des données:
    #-----------------------------------

    elif pathname == '/preprocessing':
     return html.Div([
        dbc.Container([
            # Boutons principaux
            dbc.Row([
                dbc.Col(dbc.Button("Valeurs Manquantes", id='btn-missing', color="primary", style={'width': '100%'}), width=2),
                dbc.Col(dbc.Button("Nettoyage", id='btn-replace', color="secondary", style={'width': '100%'}), width=2),
                dbc.Col(dbc.Button("Conversion Types", id='btn-convert', color="primary", style={'width': '100%'}), width=2),
                dbc.Col(dbc.Button("Normalisation", id='btn-normalize', color="secondary", style={'width': '100%'}), width=2),
                dbc.Col(dbc.Button("Suppression Doublons", id='btn-deduplicate', color="primary", style={'width': '100%'}), width=3),
            ], className="mb-3", justify="around"),
            
            # Boutons d'application cachés
            html.Div([
                dbc.Button("Confirmer Nettoyage", id='btn-confirm-replace', color="success", className="mt-2", style={'display': 'none'}),
                dbc.Button("Confirmer Conversion", id='btn-confirm-convert', color="success", className="mt-2", style={'display': 'none'}),
                dbc.Button("Confirmer Normalisation", id='btn-confirm-normalize', color="success", className="mt-2", style={'display': 'none'}),
                dbc.Button("Confirmer Dédoublonnage", id='btn-confirm-deduplicate', color="success", className="mt-2", style={'display': 'none'})
            ], id='confirmation-buttons'),
            
            # Section Téléchargement
            dbc.Card(
                [
                    dbc.CardHeader("Téléchargement des Données Modifiées"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Input(
                                                id='download-filename',
                                                type='text',
                                                placeholder='Nom du dossier/fichier',
                                                className='form-control',
                                                maxLength=50,
                                                minLength=3
                                            ),
                                            dbc.FormText(
                                                "Exemple: 'ma_analyse_2024' (sans espaces ni caractères spéciaux)",
                                                color="secondary"
                                            )
                                        ],
                                        width=8
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "💾 Télécharger",
                                            id='btn-download',
                                            color="success",
                                            className='w-100',
                                            disabled=True
                                        ),
                                        width=4
                                    )
                                ],
                                className='g-2'
                            ),
                            html.Div(id='download-status', className='mt-2'),
                            dcc.Download(id='download-trigger')
                        ]
                    )
                ],
                className='mt-4',
                id='download-card',
                style={'display': 'none'}
            ),
            
            # Zone d'affichage
            html.Div(id='preprocessing-output', className="p-4")
        ], fluid=True)
    ])
    
    #--------------------------------
    #page de visualisation des données 
    #--------------------------------

    elif pathname == '/visualization':
     # Préparation des données
     categorical_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object']
     numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
     return html.Div([
        dbc.Container(fluid=True, children=[
             
            # Première ligne : 2 visualisations côte à côte
            dbc.Row([
                # Visualisation 1: Analyse Qualitative
                dbc.Col(md=6, children=[
                    dbc.Card([
                        dbc.CardHeader("Analyse Qualitative", className="bg-info text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='quali-var1',
                                        options=[{'label': col, 'value': col} for col in categorical_cols],
                                        value=categorical_cols[0] if categorical_cols else None,
                                        clearable=False
                                    )
                                ], width=12),
                            ]),
                            dbc.RadioItems(
                                id='quali-chart-type',
                                options=[
                                    {'label': 'Diagramme en barres', 'value': 'bar'},
                                    {'label': 'Camembert', 'value': 'pie'},
                                ],
                                value='bar',
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='quali-chart',
                                style={'height': '400px'},
                                config={'displayModeBar': True}
                            )
                        ])
                    ], style={'height': '100%', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'})
                ]),
                
                # Visualisation 2: Analyse Quantitative
                dbc.Col(md=6, children=[
                    dbc.Card([
                        dbc.CardHeader("Analyse Quantitative", className="bg-info text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='quanti-var',
                                        options=[{'label': col, 'value': col} for col in numerical_cols],
                                        value=numerical_cols[0] if numerical_cols else None,
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Input(
                                        id='num-bins',
                                        type='number',
                                        placeholder='Nombre de classes',
                                        value=5,
                                        min=2,
                                        max=20
                                    )
                                ], width=6),
                            ]),
                            dbc.RadioItems(
                                id='quanti-chart-type',
                                options=[
                                    {'label': 'Histogramme', 'value': 'hist'},
                                    {'label': 'Boxplot', 'value': 'box'},
                                ],
                                value='hist',
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='quanti-chart',
                                style={'height': '400px'},
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                }
                            )
                        ])
                    ], style={'height': '100%', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'})
                ])
            ], className="mb-4"),
            # Deuxième ligne : 2 autres visualisations
            dbc.Row([
                # Visualisation 3: Analyse Mixte
                dbc.Col(md=6, children=[
                    dbc.Card([
                        dbc.CardHeader("Analyse Mixte", className="bg-info text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='mixed-quali-var',
                                        options=[{'label': col, 'value': col} for col in categorical_cols],
                                        value=categorical_cols[0] if categorical_cols else None,
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='mixed-quanti-var',
                                        options=[{'label': col, 'value': col} for col in numerical_cols],
                                        value=numerical_cols[0] if numerical_cols else None,
                                        clearable=False
                                    )
                                ], width=6),
                            ]),
                            dbc.RadioItems(
                                id='mixed-chart-type',
                                options=[
                                    {'label': 'Boxplot groupé', 'value': 'box'},
                                    {'label': 'Strip plot', 'value': 'strip'},
                                    {'label': 'Barplot', 'value': 'bar'},
                                ],
                                value='box',
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='mixed-chart',
                                style={'height': '400px'},
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                }
                            )
                        ])
                    ], style={'height': '100%', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'})
                ]),
                
                # Visualisation 4: Matrice de corrélation
                dbc.Col(md=6, children=[
                    dbc.Card([
                        dbc.CardHeader("Matrice de Corrélation", className="bg-info text-white"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='corr-vars',
                                options=[{'label': col, 'value': col} for col in numerical_cols],
                                multi=True,
                                value=numerical_cols[:4] if len(numerical_cols) > 3 else numerical_cols,
                                clearable=False
                            ),
                            dbc.Switch(
                                id='corr-annot',
                                label="Afficher les valeurs",
                                value=True,
                                className="mt-2"
                            ),
                            dcc.Graph(
                                id='correlation-chart',
                                style={'height': '400px'},
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True
                                }
                            )
                        ])
                    ], style={'height': '100%', 'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'})
                ])
            ], className="mb-4"),
            
            # Troisième ligne : Distribution des Variables (modifiée)
            dbc.Row([
                dbc.Col(md=12, children=[
                    dbc.Card([
                        dbc.CardHeader("Analyse de Distribution", className="bg-info text-white"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='dist-var',
                                        options=[{'label': col, 'value': col} for col in numerical_cols],
                                        value=numerical_cols[0] if numerical_cols else None,
                                        clearable=False
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.RadioItems(
                                        id='dist-type',
                                        options=[
                                            {'label': 'Densité', 'value': 'kde'},
                                            {'label': 'QQ-Plot', 'value': 'qq'},
                                        ],
                                        value='kde',
                                        inline=True
                                    )
                                ], width=8),
                            ]),
                            dcc.Graph(
                                id='distribution-chart',
                                style={'height': '500px'},  # Espace agrandi
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                }
                            )
                        ])
                    ], style={
                        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                        'height': '100%'
                    })
                ])
            ])
        ]),
        
        # Store pour partager des données entre callbacks
        dcc.Store(id='filtered-data'),
        
        # CSS personnalisé
        html.Div(style={'margin-bottom': '100px'})  # Espace en bas pour éviter le chevauchement
    ], style={
        'background-color': '#f8f9fa',
        'padding-top': '20px',
        'padding-bottom': '50px'
    })

    #-------------------
    #page des tests 
    #-------------------

    elif pathname == '/tests':
        return html.Div([
            dbc.Container([  # Conteneur pour organiser les éléments
            # Groupe de boutons alignés horizontalement pour chaque test
            dbc.Row([
                dbc.Col(dbc.Button("Test de Normalité", id="btn-normality", color="primary", style={'width': '100%'}), width=3),
                dbc.Col(dbc.Button("Test de Corrélation", id="btn-correlation", color="secondary", style={'width': '100%'}), width=3),
                dbc.Col(dbc.Button("Test Chi-carré", id="btn-chi-squared", color="primary", style={'width': '100%'}), width=3),
                dbc.Col(dbc.Button("Test t de Student", id="btn-t-student", color="secondary", style={'width': '100%'}), width=3),
            ], className="mb-4"),

            # Zone d'affichage des résultats des tests
            html.Div(id="test-results-content")
        ], fluid=True)
    ])
    # Partie HTML de l'interface
    elif pathname == '/notebook':
     return html.Div([
        dbc.Container([
            html.H2("✍️ Bloc-notes", className="mb-4 text-primary text-center"),
            dbc.Card([
                dbc.CardHeader("Vos Notes et Remarques", className="bg-light"),
                dbc.CardBody([
                    dcc.Textarea(
                        id='notes-textarea',
                        placeholder='Écrivez vos notes ici...',
                        style={
                            'width': '100%', 
                            'height': 300,
                            'border': '1px solid #ddd',
                            'padding': '10px',
                            'fontSize': '16px'
                        },
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Exporter en TXT", id="export-txt-btn", color="primary", className="w-100"),
                            width=2
                        ),
                        dbc.Col(
                            html.Div(id='notes-status', className="text-muted"),
                            width=6
                        )
                    ]),
                ])
            ]),
            dcc.Store(id='notes-store'),  # Pour stocker les notes localement
            dcc.Download(id='download-txt')  # Pour le téléchargement TXT
        ])
    ])



# Add this callback to synchronize the DataTable edits with the store
@app.callback(
    Output('conversion-data-store', 'data'),
    [Input('btn-convert', 'n_clicks')],
    [State('store-data', 'data')],
    prevent_initial_call=True
)
def initialize_conversion_data(n_clicks, stored_data):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    # Create basic version of data for the table
    conversion_data = [{'variable': col, 'current_type': str(df[col].dtype), 'new_type': str(df[col].dtype)} 
                    for col in df.columns]
    return conversion_data

def make_hologram_card(mode_id, emoji, title, color):
    return dbc.Col(
        dbc.Card(className="hologram-card", children=[
            html.Div(className="hologram-overlay", style={"background": color}),
            dbc.CardBody([
                html.Div(emoji, className="hologram-emoji"),
                html.H4(title, className="hologram-title")
            ])
        ], id={"type": "hologram-btn", "index": mode_id}),
        width=3, className="hologram-col"
    )

@app.callback(
    [Output('store-data', 'data'),  # Réinitialiser ou mettre à jour les données dans le store
     Output('output-data-table', 'children'),  # Mettre à jour la table
     Output('output-message', 'children')],  # Mettre à jour le message
    [Input('upload-data', 'contents'),  # Gestion du téléchargement de fichier
     Input('reset-btn', 'n_clicks')],  # Action sur le bouton "Réinitialiser"
    State('upload-data', 'filename'),  # État pour récupérer le nom du fichier
    prevent_initial_call=True  # Empêche l'exécution lors du démarrage
)

def handle_upload_and_reset(contents, reset_clicks, filename):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-btn' and reset_clicks:
        global global_df
        global_df = None
        return None, "", "Les données ont été réinitialisées."

    if triggered_id == 'upload-data' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(decoded))
            elif filename.endswith('.txt'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t')
            else:
                return None, "", "Format de fichier non supporté."

            # Format numeric values before storing
            df = format_numeric_values(df)
            
            return (
                df.to_dict('records'),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{
                        'name': col,
                        'id': col,
                        'type': 'numeric',
                        'format': dash_table.Format.Format(
                            precision=2,
                            scheme=dash_table.Format.Scheme.fixed
                        ) if pd.api.types.is_numeric_dtype(df[col]) else {}
                    } for col in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'}
                ),
                "Fichier chargé avec succès!"
            )

        except Exception as e:
            return None, "", f"Erreur lors du chargement: {str(e)}"

    return dash.no_update, dash.no_update, dash.no_update

#---------------------------------------------------------------
# Callback pour filtrer les variables en fonction de la recherche
#----------------------------------------------------------------

@app.callback(
    Output('filtered-table', 'children'),
    Input('search-input', 'value'),
    State('store-data', 'data')  # Utilisation du composant dcc.Store pour récupérer les données
)
def filter_table(search_value, stored_data):
    if search_value is None or not stored_data:
        return ""  # Si aucune donnée ou aucune recherche, on ne fait rien
    
    # Récupération des données stockées dans le store
    global_df = pd.DataFrame(stored_data)

    # Filtrage des variables par le nom (en fonction de la recherche)
    filtered_df = global_df.loc[:, global_df.columns.str.contains(search_value, case=False)]
    
    if filtered_df.empty:
        return html.Div("Aucune variable trouvée.")  # Si aucun résultat, affiche un message

    # Retourne une table avec les résultats filtrés
    return dash_table.DataTable(
        data=filtered_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in filtered_df.columns],
        style_table={'overflowX': 'auto', 'margin': '0 auto', 'width': '80%'},  # Style de la table
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '14px', 'border': '1px solid #ddd'},
        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold', 'textAlign': 'center'},
        page_size=10
    )

@app.callback(
    Output('url', 'pathname'),
    Input('summary-btn', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_summary(n_clicks):
    if n_clicks:
        return '/summary'
    return '/'

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#partie prétraitement des données 
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Callback pour afficher les interfaces
@app.callback(
    [Output('preprocessing-output', 'children', allow_duplicate=True),
     Output('confirmation-buttons', 'children', allow_duplicate=True)],
    [Input('btn-missing', 'n_clicks'),
     Input('btn-replace', 'n_clicks'),
     Input('btn-convert', 'n_clicks'),
     Input('btn-normalize', 'n_clicks'),
     Input('btn-deduplicate', 'n_clicks')],
    [State('store-data', 'data')],
    prevent_initial_call=True
)

def show_preprocessing_interface(btn_missing, btn_replace, btn_convert, btn_normalize, 
                                btn_deduplicate, stored_data):
    ctx = callback_context
    if not stored_data:
        return dbc.Alert("Veuillez d'abord charger des données", color='danger'), None

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.DataFrame(stored_data)
    output_content = html.Div()
    confirmation_button = None

    if triggered_id == 'btn-missing':
     missing = df.isnull().sum()
     missing = missing[missing > 0].sort_values(ascending=False)
    
     if missing.empty:
        output_content = dbc.Alert("Aucune valeur manquante détectée.", color="success")
     else:
        missing_df = pd.DataFrame({
            "Variable": missing.index,
            "Valeurs manquantes": missing.values.astype(int),  # Conversion en entier
            "% Manquant": (missing / len(df) * 100).round(2),
            "Type": [str(df[col].dtype) for col in missing.index]
        })
        
        # Création du graphique avec des valeurs entières
        fig = px.bar(
            missing_df,
            x='Variable',
            y='Valeurs manquantes',
            title='Répartition des valeurs manquantes',
            hover_data=['% Manquant', 'Type'],
            text='Valeurs manquantes',  # Affiche les valeurs sur les barres
            color='% Manquant',
            color_continuous_scale='Blues'
        )
        
        # Personnalisation avancée
        fig.update_traces(
            textposition='outside',  # Position du texte
            textfont_size=12,        # Taille du texte
            hovertemplate="<b>%{x}</b><br>" +
                          "Valeurs manquantes: %{y}<br>" +
                          "% Manquant: %{customdata[0]:.2f}%<br>" +
                          "Type: %{customdata[1]}<extra></extra>"
        )
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Nombre de valeurs manquantes",
            yaxis_tickformat=',d',  # Format entier pour l'axe Y
            hovermode="x unified",
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            coloraxis_colorbar=dict(title="% Manquant")
        )
        
        output_content = dbc.Card([
            dbc.CardHeader(
        html.H5("Analyse des valeurs manquantes", className="text-primary")
    ),
            dbc.CardBody([
                html.Div([
                    dcc.Graph(figure=fig, config={'displayModeBar': True}),
                    html.Hr(),
                    html.H5("Détails des valeurs manquantes:", className="text-primary"),
                    dash_table.DataTable(
                        data=missing_df.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in missing_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ])

    elif triggered_id == 'btn-replace':
     missing_count = df.isnull().sum()
     numeric_missing = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()].tolist()
     categorical_missing = df.select_dtypes(include=['object', 'category']).columns[df.select_dtypes(include=['object', 'category']).isnull().any()].tolist()
     all_missing = df.columns[df.isnull().any()].tolist()
    
     if not all_missing:
        output_content = dbc.Alert("Aucune valeur manquante à remplacer.", color="info")
        return output_content, None
     
    # Préparation des options pour chaque méthode de remplacement
     method_boxes = html.Div([
        # Titres séparés avec espace
        dbc.Row([
            dbc.Col(html.H5("Pour les variables quantitatives", className="text-primary text-center mb-4"), width=9),
            dbc.Col(html.H5("Pour les variables qualitatives", className="text-primary text-center mb-4"), width=3),
        ]),

        # Bloc des méthodes avec séparateurs
        dbc.Row([
            # KNN - Bloc 1
            dbc.Col([
                html.Div([
                    html.H5("Imputation KNN", className="text-center"),
                    dcc.Checklist(
                        id='replace-knn',
                        options=[{'label': col, 'value': col} for col in numeric_missing],
                        value=[],
                        labelStyle={'display': 'block'}
                    ),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Nombre de voisins:"),
                                dcc.Input(
                                    id='knn-n-neighbors',
                                    type='number',
                                    value=5,
                                    min=1,
                                    max=20,
                                    step=1
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Agrégation:"),
                                dcc.Dropdown(
                                    id='knn-aggregation',
                                    options=[
                                        {'label': 'Moyenne', 'value': 'mean'},
                                        {'label': 'Médiane', 'value': 'median'}
                                    ],
                                    value='mean'
                                )
                            ], width=6)
                        ], className="mt-2"),
                        html.P("Utilise la métrique 'nan_euclidean', optimisée pour les données avec valeurs manquantes.",
                               className="text-muted small mt-2")
                    ], className="mt-3", style={'border': '1px solid #eee', 'padding': '10px', 'borderRadius': '5px'})
                ], style={'border-right': '1px solid #ddd', 'padding-right': '15px', 'height': '100%'})  # Séparateur 1
            ], width=3, style={'padding-left': '0'}),  # Supprime le padding gauche pour alignement

            # Moyenne - Bloc 2
            dbc.Col([
                html.Div([
                    html.H5("Remplacer par la moyenne", className="text-center"),
                    dcc.Checklist(
                        id='replace-mean',
                        options=[{'label': col, 'value': col} for col in numeric_missing],
                        value=[],
                        labelStyle={'display': 'block'}
                    )
                ], style={'border-right': '1px solid #ddd', 'padding-right': '15px', 'height': '100%'})  # Séparateur 2
            ], width=3),

            # Zéro - Bloc 3
            dbc.Col([
                html.Div([
                    html.H5("Remplacer par zéro", className="text-center"),
                    dcc.Checklist(
                        id='replace-zero',
                        options=[{'label': col, 'value': col} for col in numeric_missing],
                        value=[],
                        labelStyle={'display': 'block'}
                    )
                ], style={'border-right': '1px solid #ddd', 'padding-right': '15px', 'height': '100%'})  # Séparateur 3
            ], width=3),

            # Mode (qualitatives) - Bloc 4
            dbc.Col([
                html.H5("Remplacer par le mode", className="text-center"),
                dcc.Checklist(
                    id='replace-mode',
                    options=[{'label': col, 'value': col} for col in categorical_missing],
                    value=[],
                    labelStyle={'display': 'block'}
                ),
                html.P("La valeur la plus fréquente (mode) sera utilisée pour chaque variable sélectionnée.",
                       className="text-muted small mt-2")
            ], width=3)
        ], className="mb-4 g-0")  # g-0 supprime les gutters pour un alignement parfait
    ])

     output_content = dbc.Card([
        dbc.CardHeader(
            html.H5("Options de nettoyage des valeurs manquantes", className="text-primary")
        ),
        dbc.CardBody([
            dbc.Alert(
                html.Div([
                    html.I(className="fas fa-info-circle me-2"),
                    "Sélectionnez une méthode de traitement pour chaque type de variable"
                ]),
                color="info"
            ),
            method_boxes,
            html.Div(id='replace-validation', className="text-danger mb-3"),
            html.Div(id='replace-preview', className="mt-3"),
            html.Div(id='replace-confirm-button-container')
        ])
    ])
     
     return output_content, None

    elif triggered_id == 'btn-convert':
     type_options = [
        {'label': 'Texte', 'value': 'object'},
        {'label': 'Nombre', 'value': 'float64'},
        {'label': 'Entier', 'value': 'int64'},
        {'label': 'Catégorie', 'value': 'category'},
        {'label': 'Date/Heure', 'value': 'datetime64[ns]'}
     ]
    
     # Créer une version de base des données pour la table
     conversion_data = [{'variable': col, 'current_type': str(df[col].dtype), 'new_type': str(df[col].dtype)} 
                      for col in df.columns]
     
     # Store the conversion data in the store component
     if 'conversion-data-store' in [p['id'] for p in dash.callback_context.outputs_list]:
         # This is a hack to update the store
         dash.callback_context.outputs_list.append({
             'id': 'conversion-data-store', 
             'property': 'data', 
             'value': conversion_data
         })
     
     # Create the conversion interface
     output_content = html.Div([
        dbc.Card([
            dbc.CardHeader(
        html.H5("Conversion des types de données", className="text-primary")
    ),
            dbc.CardBody([
                # Create the table with the provided data
                dash_table.DataTable(
                    id='type-conversion-table',
                    columns=[
                        {'name': 'Variable', 'id': 'variable', 'editable': False},
                        {'name': 'Type actuel', 'id': 'current_type', 'editable': False},
                        {'name': 'Nouveau type', 'id': 'new_type', 'presentation': 'dropdown', 'editable': True}
                    ],
                    data=conversion_data,
                    dropdown={
                        'new_type': {
                            'options': [
                                {'label': opt['label'], 'value': opt['value']} 
                                for opt in type_options
                            ]
                        }
                    },
                    editable=True,
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
                html.Div([
                    html.P("Instructions:", className="mt-3 fw-bold"),
                    html.Ul([
                        html.Li("Sélectionnez le nouveau type pour chaque variable dans la colonne 'Nouveau type'"),
                        html.Li("Les variables numériques peuvent être converties en entier (int64) ou en texte (object)"),
                        html.Li("Les variables textuelles peuvent être converties en catégories pour économiser la mémoire"),
                        html.Li("Attention: La conversion vers un type incompatible peut générer des erreurs ou des valeurs manquantes")
                    ], className="text-muted")
                ]),
                # Button to confirm conversion
                dbc.Button(
                    "Confirmer Conversion", 
                    id='btn-confirm-convert', 
                    color="primary",
                    className="mt-3"
                )
            ])
        ])
     ])
     
     return output_content, None
    
    elif triggered_id == 'btn-normalize':
     output_content = create_normalization_interface(pd.DataFrame(stored_data))
     return output_content, None  # Ajout de None pour la deuxième sortie

    elif triggered_id == 'btn-deduplicate':
     dup_count = df.duplicated().sum()
    
     output_content = dbc.Card([
        dbc.CardHeader(
        html.H5(f"Suppression des doublons ({dup_count} trouvés)", className="text-primary")
    ),
        dbc.CardBody([
            dcc.Dropdown(
                id='deduplicate-cols-select',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True,
                placeholder="Sélectionnez des colonnes (toutes par défaut)"
            ),
            dbc.RadioItems(
                id='deduplicate-keep',
                options=[
                    {'label': 'Garder la première occurrence', 'value': 'first'},
                    {'label': 'Supprimer tous les doublons', 'value': False}
                ],
                value='first'
            ),
            dbc.Button(
                "Confirmer Dédoublonnage", 
                id='btn-execute-deduplication', 
                color="primary",
                className="mt-3"
            )
        ])
    ])
    return output_content, None
    
# New callback to handle the actual validation and preview when user interacts with options
# Callback pour afficher les modes des variables qualitatives
@app.callback(
    [Output('modes-preview', 'children'),
     Output('collapse-modes', 'is_open')],
    [Input('btn-show-modes', 'n_clicks')],
    [State('store-data', 'data')],
    prevent_initial_call=True
)
def show_modes(n_clicks, stored_data):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    
    # Get the replace-mode value directly from the callback context if it exists
    mode_cols = None
    for p in dash.callback_context.inputs_list:
        if p['id'] == 'replace-mode':
            mode_cols = p['value']
            break
    
    if not mode_cols:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    modes_data = []
    
    for col in mode_cols:
        mode_value = df[col].mode().values
        if len(mode_value) > 0:
            mode_value = mode_value[0]
            modes_data.append({
                'Variable': col,
                'Mode': str(mode_value),
                'Fréquence': (df[col] == mode_value).sum(),
                '%': f"{(df[col] == mode_value).mean() * 100:.1f}%"
            })
    
    if not modes_data:
        return dbc.Alert("Aucun mode trouvé pour les variables sélectionnées.", color="warning"), True
    
    modes_table = dash_table.DataTable(
        data=modes_data,
        columns=[{'name': col, 'id': col} for col in modes_data[0].keys()],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '5px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
    
    return modes_table, True

# Callback pour la validation et l'aperçu du nettoyage (mis à jour)
@app.callback(
    [Output('replace-validation', 'children'),
     Output('replace-preview', 'children'),
     Output('replace-confirm-button-container', 'children')],
    [Input('replace-mean', 'value'),
     Input('replace-knn', 'value'),
     Input('replace-zero', 'value'),
     Input('replace-mode', 'value'),
     Input('knn-n-neighbors', 'value'),
     Input('knn-aggregation', 'value')],
    [State('store-data', 'data')],
    prevent_initial_call=True
)
def update_preview_on_selection(mean_cols, knn_cols, zero_cols, mode_cols, knn_neighbors, knn_aggregation, stored_data):
    if not stored_data:
        raise PreventUpdate
    
    # Initialize all variables as lists if they are None
    mean_cols = mean_cols or []
    knn_cols = knn_cols or []
    zero_cols = zero_cols or []
    mode_cols = mode_cols or []
    knn_neighbors = knn_neighbors or 5
    knn_aggregation = knn_aggregation or 'mean'
    
    # Now we can safely concatenate all lists
    all_selected = mean_cols + knn_cols + zero_cols + mode_cols
    
    df = pd.DataFrame(stored_data)
    validation_msg = ""
    preview_content = html.Div()
    
    # Vérification des conflits
    duplicates = [col for col in all_selected if all_selected.count(col) > 1]
    
    if duplicates:
        validation_msg = f"Erreur: Variables sélectionnées dans plusieurs méthodes: {', '.join(set(duplicates))}"
    
    # Vérification des variables non traitées
    missing_cols = df.columns[df.isnull().any()].tolist()
    untreated = set(missing_cols) - set(all_selected)
    
    if untreated and not duplicates:
        validation_msg = f"Attention: Variables sans méthode sélectionnée: {', '.join(untreated)}"
    
    # Prévisualisation des modifications
    if not validation_msg and (mean_cols or knn_cols or zero_cols or mode_cols):
        preview_df = df.copy()
        modified_rows = set()
        preview_data = []
        
        # Traitement par la moyenne
        for col in mean_cols:
            na_indices = preview_df[col].isnull()
            mean_val = preview_df[col].mean()
            preview_df.loc[na_indices, col] = mean_val
            modified_rows.update(preview_df.index[na_indices].tolist())
            
            # Ajout à la prévisualisation
            for idx in preview_df.index[na_indices][:3]:  # Limité à 3 exemples par colonne
                preview_data.append({
                    'Variable': col,
                    'Index': idx,
                    'Avant': 'NA',
                    'Après': f"{preview_df.loc[idx, col]:.2f}",
                    'Méthode': 'Moyenne'
                })
        
        # Traitement par zéro
        for col in zero_cols:
            na_indices = preview_df[col].isnull()
            preview_df.loc[na_indices, col] = 0
            modified_rows.update(preview_df.index[na_indices].tolist())
            
            for idx in preview_df.index[na_indices][:3]:
                preview_data.append({
                    'Variable': col,
                    'Index': idx,
                    'Avant': 'NA',
                    'Après': "0",
                    'Méthode': 'Zéro'
                })
        
        # Traitement par le mode (variables qualitatives)
        for col in mode_cols:
            na_indices = preview_df[col].isnull()
            if na_indices.any():
                mode_val = preview_df[col].mode()
                if not mode_val.empty:
                    mode_val = mode_val.iloc[0]
                    preview_df.loc[na_indices, col] = mode_val
                    modified_rows.update(preview_df.index[na_indices].tolist())
                    
                    for idx in preview_df.index[na_indices][:3]:
                        preview_data.append({
                            'Variable': col,
                            'Index': idx,
                            'Avant': 'NA',
                            'Après': str(mode_val),
                            'Méthode': 'Mode'
                        })
        
        # Traitement KNN (seulement numérique)
        if knn_cols:
            numeric_cols = preview_df.select_dtypes(include=['number']).columns.tolist()
            knn_cols = [col for col in knn_cols if col in numeric_cols]
            
            if knn_cols:
                try:
                    numeric_data = preview_df[numeric_cols].copy()
                    imputer = KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
                    imputed_values = imputer.fit_transform(numeric_data)
                    
                    for i, col in enumerate(numeric_cols):
                        if col in knn_cols:
                            na_indices = preview_df[col].isnull()
                            preview_df.loc[na_indices, col] = imputed_values[na_indices.values, i]
                            modified_rows.update(preview_df.index[na_indices].tolist())
                            
                            for idx in preview_df.index[na_indices][:3]:
                                preview_data.append({
                                    'Variable': col,
                                    'Index': idx,
                                    'Avant': 'NA',
                                    'Après': f"{preview_df.loc[idx, col]:.2f}",
                                    'Méthode': 'KNN'
                                })
                except Exception as e:
                    validation_msg = f"Erreur KNN (numérique): {str(e)}"
        
        # Création de l'aperçu
        if preview_data:
            preview_content = html.Div([
                html.H5("Aperçu des modifications"),
                dash_table.DataTable(
                    data=preview_data,
                    columns=[{'name': col, 'id': col} for col in preview_data[0].keys()],
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_data_conditional=[
                        {'if': {'column_id': 'Après'}, 'backgroundColor': '#e6f3ff', 'fontWeight': 'bold'}
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    page_size=10
                )
            ])
    
    # Bouton de confirmation
    confirm_button = dbc.Button(
        "Confirmer le remplacement",
        id='btn-confirm-replace',
        color="primary",
        className="mt-3",
        disabled=bool(validation_msg)  # Désactivé s'il y a des erreurs
    ) if (mean_cols or knn_cols or zero_cols or mode_cols) else None
    
    return validation_msg, preview_content, confirm_button

# Callback pour appliquer le nettoyage (mis à jour pour inclure le mode)
@app.callback(
    [Output('store-data', 'data', allow_duplicate=True),
     Output('preprocessing-output', 'children', allow_duplicate=True)],
    [Input('btn-confirm-replace', 'n_clicks')],
    [State('store-data', 'data'),
     State('replace-mean', 'value'),
     State('replace-knn', 'value'),
     State('replace-zero', 'value'),
     State('replace-mode', 'value'),
     State('knn-n-neighbors', 'value'),
     State('knn-aggregation', 'value')],
    prevent_initial_call=True
)
def apply_cleaning(n_clicks, stored_data, mean_cols, knn_cols, zero_cols, mode_cols, knn_neighbors, knn_aggregation):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    
    # Initialize all variables as lists if they are None
    mean_cols = mean_cols or []
    knn_cols = knn_cols or []
    zero_cols = zero_cols or []
    mode_cols = mode_cols or []
    knn_neighbors = knn_neighbors or 5
    knn_aggregation = knn_aggregation or 'mean'
    
    # Vérifier si aucune méthode n'a été sélectionnée
    if not any([mean_cols, knn_cols, zero_cols, mode_cols]):
        return stored_data, html.Div("Veuillez sélectionner au moins une méthode de remplacement.", className="alert alert-warning")
    
    df_original = pd.DataFrame(stored_data)
    df = df_original.copy()
    
    # Dictionaries to store changes for each method
    mean_changes = []
    knn_changes = []
    zero_changes = []
    mode_changes = []
    
    # Apply mean replacement
    for col in mean_cols:
        if col in df.columns and df[col].isna().any():
            missing_before = df[col].isna().sum()
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            missing_after = df[col].isna().sum()
            if missing_before > missing_after:
                mean_changes.append({
                    'Variable': col,
                    'Valeurs manquantes avant': missing_before,
                    'Valeurs manquantes après': missing_after,
                    'Moyenne utilisée': f"{mean_value:.2f}"
                })
    
    # Apply KNN imputation
    if knn_cols:
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            knn_cols = [col for col in knn_cols if col in numeric_cols]
            
            if knn_cols:
                missing_before_dict = {col: df[col].isna().sum() for col in knn_cols}
                
                imputer = KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
                df[knn_cols] = imputer.fit_transform(df[knn_cols])
                
                for col in knn_cols:
                    missing_after = df[col].isna().sum()
                    if missing_before_dict[col] > missing_after:
                        knn_changes.append({
                            'Variable': col,
                            'Valeurs manquantes avant': missing_before_dict[col],
                            'Valeurs manquantes après': missing_after,
                            'Paramètres KNN': f"k={knn_neighbors}, {knn_aggregation}"
                        })
        except Exception as e:
            return stored_data, html.Div(f"Erreur lors de l'imputation KNN : {str(e)}", className="alert alert-danger")
    
    # Apply zero replacement
    for col in zero_cols:
        if col in df.columns and df[col].isna().any():
            missing_before = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            missing_after = df[col].isna().sum()
            if missing_before > missing_after:
                zero_changes.append({
                    'Variable': col,
                    'Valeurs manquantes avant': missing_before,
                    'Valeurs manquantes après': missing_after
                })
    
    # Apply mode replacement
    for col in mode_cols:
        if col in df.columns and df[col].isna().any():
            missing_before = df[col].isna().sum()
            mode_value = df[col].mode().values
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
                missing_after = df[col].isna().sum()
                if missing_before > missing_after:
                    mode_changes.append({
                        'Variable': col,
                        'Valeurs manquantes avant': missing_before,
                        'Valeurs manquantes après': missing_after,
                        'Mode utilisé': str(mode_value[0])
                    })
    
    # Create the summary tables
    tables = []
    
    if mean_changes:
        tables.append(html.Div([
            html.H5("Remplacer par la moyenne", className="text-primary"),
            dash_table.DataTable(
                data=mean_changes,
                columns=[{'name': col, 'id': col} for col in mean_changes[0].keys()],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ]))
    
    if knn_changes:
        tables.append(html.Div([
            html.H5("Imputation KNN", className="text-primary"),
            dash_table.DataTable(
                data=knn_changes,
                columns=[{'name': col, 'id': col} for col in knn_changes[0].keys()],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ]))
    
    if zero_changes:
        tables.append(html.Div([
            html.H5("Remplacer par zéro", className="text-primary"),
            dash_table.DataTable(
                data=zero_changes,
                columns=[{'name': col, 'id': col} for col in zero_changes[0].keys()],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ]))
    
    if mode_changes:
        tables.append(html.Div([
            html.H5("Remplacer par le mode", className="text-primary"),
            dash_table.DataTable(
                data=mode_changes,
                columns=[{'name': col, 'id': col} for col in mode_changes[0].keys()],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ]))
    
    if not any([mean_changes, knn_changes, zero_changes, mode_changes]):
        return stored_data, html.Div("Aucune valeur manquante n'a été trouvée dans les colonnes sélectionnées.", className="alert alert-info")
    
    # Create the final summary
    total_missing_before = df_original.isna().sum().sum()
    total_missing_after = df.isna().sum().sum()
    
    summary = html.Div([
        html.H4("Résumé des modifications", className="text-primary mb-4"),
        html.Div([
            html.P(f"Valeurs manquantes avant : {total_missing_before}"),
            html.P(f"Valeurs manquantes après : {total_missing_after}"),
            html.P(f"Taux de succès : {100 - (total_missing_after / total_missing_before * 100 if total_missing_before > 0 else 0):.2f}%")
        ], className="alert alert-success mb-4")
    ])
    
    # Combine all tables
    result_content = html.Div([
        summary,
        *tables
    ])
    
    return df.to_dict('records'), result_content

#-------------------------------------
# Callback pour la conversion de types
#-------------------------------------

@app.callback(
    [Output('store-data', 'data', allow_duplicate=True),
     Output('preprocessing-output', 'children', allow_duplicate=True)],
    [Input('btn-confirm-convert', 'n_clicks')],
    [State('conversion-data-store', 'data'),
     State('store-data', 'data')],
    prevent_initial_call=True
)
def apply_conversion(n_clicks, conversion_data, stored_data):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get DataFrame from stored data
    df = pd.DataFrame(stored_data)
    report = []
    
    # Check if conversion data is available
    if not conversion_data:
        # Generate conversion data on the fly if not available
        conversion_data = [{'variable': col, 'current_type': str(df[col].dtype), 'new_type': str(df[col].dtype)} 
                        for col in df.columns]
    
    # Perform conversions based on conversion_data
    for item in conversion_data:
        col = item.get('variable')
        current_type = item.get('current_type')
        new_type = item.get('new_type')
        
        # Skip if no change in type
        if current_type == new_type or not col or not new_type:
            continue
        
        try:
            # Handle different type conversions
            if new_type == 'int64':
                # Check if conversion is possible
                if df[col].isnull().any():
                    # Fill NA values with 0 before conversion to int (or could use other strategies)
                    df[col] = df[col].fillna(0)
                df[col] = df[col].astype('int64')
                report.append(f"{col}: {current_type} → {new_type} (valeurs NaN remplacées par 0)")
            
            elif new_type == 'float64':
                df[col] = df[col].astype('float64')
                report.append(f"{col}: {current_type} → {new_type}")
            
            elif new_type == 'category':
                df[col] = df[col].astype('category')
                report.append(f"{col}: {current_type} → {new_type}")
            
            elif new_type == 'object':
                df[col] = df[col].astype('object')
                report.append(f"{col}: {current_type} → {new_type}")
            
            elif new_type == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
                report.append(f"{col}: {current_type} → {new_type}")
            
        except Exception as e:
            report.append(f"Erreur conversion {col}: {str(e)}")
    
    if report:
        result_content = dbc.Card([
            dbc.CardHeader("Rapport de conversion des types"),
            dbc.CardBody([
                html.Ul([html.Li(item) for item in report]),
                html.H5("Aperçu des données converties:"),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'}
                )
            ])
        ])
    else:
        result_content = dbc.Alert("Aucune conversion effectuée.", color="warning")
    
    return df.to_dict('records'), result_content

@app.callback(
    Output('normalization-preview', 'children'),
    [Input('normalize-var-select', 'value'),
     Input('normalize-method-select', 'value')],
    [State('store-data', 'data')]
)
def update_normalization_preview(selected_var, method, stored_data):
    if not selected_var or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    preview_df = df[[selected_var]].copy()
    
    # Appliquer la normalisation temporaire pour la prévisualisation
    try:
        if method == 'standard':
            scaler = StandardScaler()
            preview_df[f'{selected_var}_norm'] = scaler.fit_transform(preview_df[[selected_var]])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            preview_df[f'{selected_var}_norm'] = scaler.fit_transform(preview_df[[selected_var]])
        elif method == 'robust':
            scaler = RobustScaler()
            preview_df[f'{selected_var}_norm'] = scaler.fit_transform(preview_df[[selected_var]])
        elif method == 'log':
            preview_df[f'{selected_var}_norm'] = np.log1p(preview_df[selected_var])
        
        # Créer les visualisations
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Avant Normalisation", "Après Normalisation"))
        
        fig.add_trace(
            go.Histogram(
                x=preview_df[selected_var],
                name='Original',
                marker_color='blue',
                nbinsx=30
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=preview_df[f'{selected_var}_norm'],
                name='Normalisé',
                marker_color='orange',
                nbinsx=30
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return html.Div([
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        data=preview_df.head().to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in preview_df.columns],
                        page_size=5,
                        style_table={'overflowX': 'auto'}
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=fig),
                    width=6
                )
            ])
        ])
    
    except Exception as e:
        return dbc.Alert(f"Erreur de prévisualisation : {str(e)}", color="danger")
    
#-------------------------------------------------------
# Modifier le callback d'application de la normalisation
#-------------------------------------------------------

@app.callback(
    [Output('store-data', 'data', allow_duplicate=True),
     Output('normalization-result', 'children')],
    [Input('btn-apply-normalization', 'n_clicks')],
    [State('normalize-var-select', 'value'),
     State('normalize-method-select', 'value'),
     State('store-data', 'data')],
    prevent_initial_call=True
)
def apply_normalization(n_clicks, selected_var, method, stored_data):
    if not n_clicks:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    new_col = f"{selected_var}_norm"
    
    try:
        if method == 'standard':
            scaler = StandardScaler()
            df[new_col] = scaler.fit_transform(df[[selected_var]])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[new_col] = scaler.fit_transform(df[[selected_var]])
        elif method == 'robust':
            scaler = RobustScaler()
            df[new_col] = scaler.fit_transform(df[[selected_var]])
        elif method == 'log':
            df[new_col] = np.log1p(df[selected_var])
        
        # Créer le rapport de modification
        stats = pd.DataFrame({
            'Statistique': ['Min', 'Max', 'Moyenne', 'Écart-type'],
            'Original': [
                df[selected_var].min(),
                df[selected_var].max(),
                df[selected_var].mean(),
                df[selected_var].std()
            ],
            'Normalisé': [
                df[new_col].min(),
                df[new_col].max(),
                df[new_col].mean(),
                df[new_col].std()
            ]
        })
        
        return df.to_dict('records'), dbc.Card([
            dbc.CardHeader("Normalisation appliquée avec succès ✅"),
            dbc.CardBody([
                html.H5(f"Nouvelle colonne créée : {new_col}", className="text-success"),
                dash_table.DataTable(
                    data=stats.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in stats.columns],
                    style_cell={'textAlign': 'center'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
                dbc.Alert("La colonne normalisée a été ajoutée à votre jeu de données.", 
                         color="success",
                         className="mt-3")
            ])
        ])
    
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Erreur lors de l'application : {str(e)}", color="danger")

# Modifier l'interface de normalisation
def create_normalization_interface(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    return dbc.Card([
        dbc.CardHeader(
        html.H5("Normalisation des variables numériques", className="text-primary")
    ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Variable à normaliser :"),
                    dcc.Dropdown(
                        id='normalize-var-select',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        value=numeric_cols[0] if numeric_cols else None,
                        clearable=False
                    )
                ], width=4),
                
                dbc.Col([
                    dbc.Label("Méthode de normalisation :"),
                    dbc.RadioItems(
                        id='normalize-method-select',
                        options=[
                            {'label': 'Standard (moyenne=0, écart-type=1)', 'value': 'standard'},
                            {'label': 'Min-Max [0-1]', 'value': 'minmax'},
                            {'label': 'Robuste (résistant aux outliers)', 'value': 'robust'},
                            {'label': 'Logarithmique', 'value': 'log'}
                        ],
                        value='standard',
                        inline=False
                    )
                ], width=8)
            ]),
            
            html.Hr(),
            
            html.H4("Aperçu de la Normalisation", className="mb-3"),
            html.Div(id='normalization-preview'),
            
            html.Hr(),
            
            dbc.Button("Appliquer la Normalisation", 
                      id='btn-apply-normalization',
                      color="primary",
                      className="mt-3"),
            
            html.Div(id='normalization-result')
        ])
    ])

#----------------------------------------------------
# Callback pour exécuter la suppression des doublons
#----------------------------------------------------

@app.callback(
    [Output('store-data', 'data', allow_duplicate=True),
     Output('preprocessing-output', 'children', allow_duplicate=True)],
    Input('btn-execute-deduplication', 'n_clicks'),
    [State('store-data', 'data'),
     State('deduplicate-cols-select', 'value'),
     State('deduplicate-keep', 'value')],
    prevent_initial_call=True
)
def execute_deduplication(n_clicks, stored_data, columns, keep):
    if not n_clicks:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    initial_count = len(df)
    
    try:
        # Utiliser toutes les colonnes si aucune sélection
        subset = columns if columns else None
        
        # Appliquer la suppression
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_clean)
        
        # Préparer le rapport
        result_content = dbc.Card([
            dbc.CardHeader(
        html.H5("Résultats du dédoublonnage", className="text-primary")
    ),
            dbc.CardBody([
                dbc.Alert(
                    f"{removed_count} doublons supprimés avec succès!",
                    color="success",
                    className="mb-3"
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(f"Lignes initiales: {initial_count}"),
                        html.Div(f"Lignes restantes: {len(df_clean)}"),
                        html.Div(f"Pourcentage supprimé: {removed_count/initial_count:.1%}")
                    ], width=6),
                    dbc.Col([
                        html.H5("Aperçu des données nettoyées:"),
                        dash_table.DataTable(
                            data=df_clean.head().to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in df_clean.columns],
                            page_size=5
                        )
                    ], width=6)
                ])
            ])
        ])
        
        return df_clean.to_dict('records'), result_content
    
    except Exception as e:
        return dash.no_update, dbc.Alert(
            f"Erreur lors de la suppression: {str(e)}",
            color="danger"
        )

#---------------------------------------------------------------------
# --------------------------------------------------------------------
# Partie Visualisation
#---------------------------------------------------------------------
#---------------------------------------------------------------------

@app.callback(
    Output('quali-chart', 'figure'),
    [Input('quali-var1', 'value'),
     Input('quali-chart-type', 'value')],
    [State('store-data', 'data')]
)
def update_quali_chart(variable, chart_type, stored_data):
    if not variable or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    counts = df[variable].value_counts().reset_index()
    counts.columns = ['category', 'count']
    
    try:
        if chart_type == 'pie':
            fig = px.pie(
                counts,
                names='category',
                values='count',
                title=f"Répartition de {variable}",
                hole=0.3,
                labels={'category': variable}
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=1)))
        else:
            fig = px.bar(
                counts,
                x='category',
                y='count',
                title=f"Distribution de {variable}",
                labels={'category': variable, 'count': 'Fréquence'},
                text='count',
                color='category'
            )
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=variable,
                yaxis_title="Fréquence"
            )
        
        fig.update_layout(
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
        
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")
        return go.Figure()

# Modifier le callback quanti-quanti pour ajouter plus d'interactivité
@app.callback(
    Output('quanti-chart', 'figure'),
    [Input('quanti-var', 'value'),
     Input('num-bins', 'value'),
     Input('quanti-chart-type', 'value')],
    [State('store-data', 'data')]
)
def update_quanti_chart(variable, n_bins, chart_type, stored_data):
    if not variable or not stored_data or not n_bins:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    data = df[variable].dropna()
    
    try:
        if chart_type == 'hist':
            # Version avec Graph Objects au lieu de Plotly Express
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=n_bins,
                marker_color='#636EFA',
                opacity=0.75
            ))
            fig.update_layout(
                title=f"Histogramme de {variable}",
                xaxis_title=variable,
                yaxis_title="Fréquence",
                bargap=0.1
            )
            
        elif chart_type == 'box':  # boxplot
            fig = px.box(
                df,
                y=variable,
                title=f"Distribution de {variable}"
            )
        else:
            raise PreventUpdate  # Empêcher la mise à jour si un type non pris en charge est sélectionné
        
        # Paramètres communs
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="closest"
        )
        
        return fig
        
    except Exception as e:
        print(f"Erreur dans update_quanti_chart: {e}")
        return go.Figure()  # Retourne une figure vide en cas d'erreur
    
@app.callback(
    Output('mixed-chart', 'figure'),
    [Input('mixed-quali-var', 'value'),
     Input('mixed-quanti-var', 'value'),
     Input('mixed-chart-type', 'value')],
    [State('store-data', 'data')]
)
def update_mixed_chart(quali_var, quanti_var, chart_type, stored_data):
    if not quali_var or not quanti_var or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    
    # Vérification que les colonnes existent
    if quali_var not in df.columns or quanti_var not in df.columns:
        return go.Figure().update_layout(
            title="Erreur: Variables non trouvées",
            annotations=[{
                'text': "Les variables sélectionnées n'existent pas dans le jeu de données",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    
    # Conversion des types si nécessaire
    try:
        df[quanti_var] = pd.to_numeric(df[quanti_var], errors='coerce')
        df[quali_var] = df[quali_var].astype(str)
    except Exception as e:
        return go.Figure().update_layout(
            title="Erreur de conversion des données",
            annotations=[{
                'text': f"Erreur lors de la conversion des données: {str(e)}",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    
    # Suppression des valeurs manquantes
    df = df.dropna(subset=[quali_var, quanti_var])
    
    # Vérification que les données ne sont pas vides après nettoyage
    if len(df) == 0:
        return go.Figure().update_layout(
            title="Pas de données disponibles",
            annotations=[{
                'text': "Aucune donnée valide après nettoyage",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    
    try:
        if chart_type == 'box':
            fig = px.box(
                df,
                x=quali_var,
                y=quanti_var,
                color=quali_var,
                title=f"Distribution de {quanti_var} par {quali_var}",
                points="all",
                notched=True
            )
            
            # Amélioration du style
            fig.update_traces(
                boxmean=True,  # Affiche la moyenne
                jitter=0.3,    # Espacement des points
                pointpos=-1.8, # Position des points
                marker=dict(size=4, opacity=0.5),
                boxpoints='all'
            )
            
            # Ajout des statistiques par groupe
            stats = df.groupby(quali_var)[quanti_var].agg(['mean', 'std', 'count']).round(2)
            stats_text = "<br>".join([
                f"<b>{group}</b>: n={row['count']}, Moyenne={row['mean']:.2f}, Écart-type={row['std']:.2f}"
                for group, row in stats.iterrows()
            ])
            
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
            
        elif chart_type == 'strip':
            fig = px.strip(
                df,
                x=quali_var,
                y=quanti_var,
                color=quali_var,
                title=f"Distribution de {quanti_var} par {quali_var}",
                hover_data=[quali_var, quanti_var]
            )
            
            # Amélioration du style
            fig.update_traces(
                jitter=0.3,
                marker=dict(size=8, opacity=0.6),
                marker_line=dict(width=1, color='white')
            )
            
            # Ajout des moyennes par groupe
            means = df.groupby(quali_var)[quanti_var].mean()
            for i, (group, mean) in enumerate(means.items()):
                fig.add_shape(
                    type="line",
                    x0=i-0.4,
                    x1=i+0.4,
                    y0=mean,
                    y1=mean,
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=i,
                    y=mean,
                    text=f"Moyenne: {mean:.2f}",
                    showarrow=False,
                    yshift=10
                )
            
        else:  # Barplot
            # Calcul des statistiques par groupe
            df_agg = df.groupby(quali_var)[quanti_var].agg(['mean', 'std', 'count']).reset_index()
            df_agg.columns = [quali_var, 'moyenne', 'ecart_type', 'effectif']
            
            fig = px.bar(
                df_agg,
                x=quali_var,
                y='moyenne',
                color=quali_var,
                title=f"Moyenne de {quanti_var} par {quali_var}",
                error_y='ecart_type',
                text='moyenne'
            )
            
            # Amélioration du style
            fig.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            
            # Ajout des effectifs
            for i, row in df_agg.iterrows():
                fig.add_annotation(
                    x=row[quali_var],
                    y=row['moyenne'],
                    text=f"n={row['effectif']}",
                    showarrow=False,
                    yshift=-30
                )
        
        # Paramètres communs
        fig.update_layout(
            xaxis_title=quali_var,
            yaxis_title=quanti_var,
            hovermode="closest",
            showlegend=False,
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            template="plotly_white",
            margin=dict(l=50, r=50, t=80, b=50),
            height=500
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().update_layout(
            title="Erreur lors de la création du graphique",
            annotations=[{
                'text': f"Erreur: {str(e)}",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )

# Modifier le callback pour la matrice de corrélation
@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('corr-vars', 'value'),
     Input('corr-annot', 'value')],
    [State('store-data', 'data')]
)
def update_correlation_matrix(selected_vars, show_annot, stored_data):
    if not selected_vars or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)[selected_vars].dropna()
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=selected_vars,
        y=selected_vars,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    if show_annot:
        fig.update_traces(
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Corrélation: %{z:.2f}<extra></extra>"
        )
    else:
        fig.update_traces(
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Corrélation: %{z:.2f}<extra></extra>"
        )
    
    fig.update_layout(
        title="Matrice de Corrélation",
        xaxis_side="top",
        margin=dict(l=0, r=0, b=0, t=40),
        coloraxis_colorbar=dict(
            title="Corrélation",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@app.callback(
    Output('distribution-chart', 'figure'),
    [Input('dist-var', 'value'),
     Input('dist-type', 'value')],
    [State('store-data', 'data')]
)
def update_distribution_chart(variable, dist_type, stored_data):
    if not variable or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    data = df[variable].dropna()
    
    # Check if data is empty after dropping NA values
    if len(data) == 0:
        return go.Figure().update_layout(
            title="Pas de données disponibles",
            annotations=[{
                'text': "Aucune donnée valide pour cette variable",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    
    try:
        if dist_type == 'kde':
            fig = go.Figure()
            
            # Add histogram with improved styling
            fig.add_trace(go.Histogram(
                x=data,
                name='Histogramme',
                opacity=0.7,
                nbinsx=30,
                marker_color='#636EFA',
                marker_line_color='white',
                marker_line_width=1,
                histnorm='probability density'
            ))
            
            # Add KDE with improved styling
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde(x_range),
                name='Densité',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Moyenne: {mean_val:.2f}",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Médiane: {median_val:.2f}",
                annotation_position="top left"
            )
            
            # Add normal distribution curve for comparison
            mu, std = stats.norm.fit(data)
            normal_curve = stats.norm.pdf(x_range, mu, std)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_curve,
                name='Distribution normale',
                line=dict(color='green', width=1, dash='dot'),
                opacity=0.5
            ))
            
            fig.update_layout(
                title=f"Distribution de {variable}",
                xaxis_title=variable,
                yaxis_title="Densité",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
        else:  # QQ-Plot
            fig = go.Figure()
            
            # Calcul des quantiles théoriques et observés
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            observed = np.sort(data)
            
            # Ajout des points du QQ-plot
            fig.add_trace(go.Scatter(
                x=theoretical,
                y=observed,
                mode='markers',
                name='Points observés',
                marker=dict(
                    color='#636EFA',
                    size=8,
                    line=dict(width=1, color='white')
                ),
                hovertemplate="Quantile théorique: %{x:.2f}<br>Quantile observé: %{y:.2f}<extra></extra>"
            ))
            
            # Ligne de référence (y = x)
            fig.add_trace(go.Scatter(
                x=[theoretical[0], theoretical[-1]],
                y=[theoretical[0], theoretical[-1]],
                mode='lines',
                name='Distribution normale',
                line=dict(color="red", dash="dash", width=2)
            ))
            
            # Mise à jour du layout avec une meilleure organisation des légendes
            fig.update_layout(
                title=dict(
                    text=f"QQ-Plot de {variable}",
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis_title="Quantiles théoriques",
                yaxis_title="Quantiles observés",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                ),
                margin=dict(t=100, r=50, l=50, b=50)
            )
        
        # Common parameters
        fig.update_layout(
            hovermode="closest",
            margin=dict(l=50, r=50, t=80, b=50),
            height=500
        )
        
        # Add descriptive statistics
        stats_text = f"""
        <b>Statistiques:</b><br>
        Moyenne: {data.mean():.2f}<br>
        Écart-type: {data.std():.2f}<br>
        Skewness: {data.skew():.2f}<br>
        Kurtosis: {data.kurtosis():.2f}
        """
        
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().update_layout(
            title="Erreur lors de la création du graphique",
            annotations=[{
                'text': f"Erreur: {str(e)}",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }]
        )

@app.callback(
    [Output('quali-var1', 'options'),
     Output('quanti-var', 'options'),    # Changé de quanti-var-x à quanti-var
     Output('mixed-quali-var', 'options'),
     Output('mixed-quanti-var', 'options'),
     Output('corr-vars', 'options'),
     Output('dist-var', 'options')],
    [Input('store-data', 'modified_timestamp')],
    [State('store-data', 'data')]
)
def update_dropdown_options(ts, stored_data):
    if not stored_data:
        return [], [], [], [], [], []
    
    df = pd.DataFrame(stored_data)
    
    # Variables qualitatives
    categorical_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object']
    # Variables quantitatives
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    quali_options = [{'label': col, 'value': col} for col in categorical_cols]
    quanti_options = [{'label': col, 'value': col} for col in numerical_cols]
    
    return (
        quali_options,
        quanti_options,  # Options pour quanti-var
        quali_options,
        quanti_options,
        quanti_options,
        quanti_options
    )

@app.callback(
    [Output('quali-var1', 'value'),
     Output('quanti-var', 'value')],    # Changé de quanti-var-x à quanti-var
    [Input('quali-var1', 'options'),
     Input('quanti-var', 'options')]    # Changé de quanti-var-x à quanti-var
)
def set_default_values(quali_opts, quanti_opts):
    # Retourner None si aucune option n'est disponible
    quali_val = quali_opts[0]['value'] if quali_opts else None
    quanti_val = quanti_opts[0]['value'] if quanti_opts else None
    
    return quali_val, quanti_val

# Callback pour exporter les graphiques
@app.callback(
    Output('export-graph', 'data'),
    [Input('export-png', 'n_clicks'),
     Input('export-svg', 'n_clicks'),
     Input('export-html', 'n_clicks')],
    [State('current-graph', 'figure')],
    prevent_initial_call=True
)
def export_graph(png_clicks, svg_clicks, html_clicks, figure):
    ctx = dash.callback_context
    if not ctx.triggered or not figure:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if trigger_id == 'export-png':
        return dcc.send_bytes(
            lambda x: pio.write_image(figure, x, format='png'),
            f"graph_{timestamp}.png"
        )
    elif trigger_id == 'export-svg':
        return dcc.send_bytes(
            lambda x: pio.write_image(figure, x, format='svg'),
            f"graph_{timestamp}.svg"
        )
    elif trigger_id == 'export-html':
        return dcc.send_bytes(
            lambda x: pio.write_html(figure, x),
            f"graph_{timestamp}.html"
        )

# Callback pour la sélection interactive
@app.callback(
    Output('selected-data', 'children'),
    [Input('quanti-quanti-chart', 'selectedData'),
     Input('mixed-chart', 'selectedData')],
    [State('store-data', 'data')]
)
def display_selected_data(quanti_selected, mixed_selected, stored_data):
    ctx = dash.callback_context
    if not ctx.triggered or not stored_data:
        raise PreventUpdate
    
    df = pd.DataFrame(stored_data)
    selected_indices = []
    
    if ctx.triggered[0]['prop_id'] == 'quanti-quanti-chart.selectedData':
        if quanti_selected and 'points' in quanti_selected:
            selected_indices = [p['pointIndex'] for p in quanti_selected['points']]
    elif ctx.triggered[0]['prop_id'] == 'mixed-chart.selectedData':
        if mixed_selected and 'points' in mixed_selected:
            selected_indices = [p['pointIndex'] for p in mixed_selected['points']]
    
    if selected_indices:
        selected_df = df.iloc[selected_indices]
        return html.Div([
            html.H5("Données sélectionnées:"),
            dash_table.DataTable(
                data=selected_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in selected_df.columns],
                page_size=5,
                style_table={'overflowX': 'auto'}
            )
        ])
    
    return html.Div("Sélectionnez des points dans les graphiques pour voir les données correspondantes")


#-----------------------------------------------
# =============================================
# partie tests
# =============================================
#-----------------------------------------------

# Ajouter ce callback pour gérer l'affichage des tests
@app.callback(
    Output("test-results-content", "children"),
    [Input("btn-normality", "n_clicks"),
     Input("btn-correlation", "n_clicks"),
     Input("btn-chi-squared", "n_clicks"),
     Input("btn-t-student", "n_clicks")],
    [State("store-data", "data")]
)
def display_test_interface(norm_clicks, corr_clicks, chi_clicks, t_clicks, data):
    ctx = dash.callback_context
    if not ctx.triggered or not data:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.DataFrame(data)
    
    # Test de Normalité
    if triggered_id == "btn-normality":
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Test de Normalité", className="text-white"),
                html.P("Vérifie si une variable numérique suit une distribution normale. Essentiel avant d'appliquer "
                      "de nombreux tests paramétriques. Choisissez entre Shapiro-Wilk (échantillons < 50) ou "
                      "Kolmogorov-Smirnov (échantillons plus grands).", className="text-dark mb-0")
            ], className="bg-info"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Variable numérique :"),
                        dcc.Dropdown(
                            id="normality-var-select",
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Type de test :"),
                        dbc.RadioItems(
                            id="normality-test-type",
                            options=[
                                {'label': 'Shapiro-Wilk (recommandé pour n < 50)', 'value': 'shapiro'},
                                {'label': 'Kolmogorov-Smirnov (pour n ≥ 50)', 'value': 'ks'}
                            ],
                            value='shapiro'
                        )
                    ], width=6)
                ]),
                dbc.Button("Lancer le test", 
                         id="btn-run-normality", 
                         color="success",
                         className="mt-3"),
                html.Div(id="normality-results", className="mt-4")
            ])
        ])
    
    # Test de Corrélation
    elif triggered_id == "btn-correlation":
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Test de Corrélation", className="text-white"),
                html.P("Mesure la relation linéaire (Pearson) ou monotone (Spearman) entre deux variables numériques. "
                      "Pearson suppose une relation linéaire et une distribution normale, tandis que Spearman est "
                      "non-paramétrique et mesure les relations monotones.", className="text-dark mb-0")
            ], className="bg-info"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Variable 1 :"),
                        dcc.Dropdown(
                            id="corr-var1-select",
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if len(numerical_cols) > 0 else None
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Variable 2 :"),
                        dcc.Dropdown(
                            id="corr-var2-select",
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[1] if len(numerical_cols) > 1 else None
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Méthode :"),
                        dbc.RadioItems(
                            id="corr-method-select",
                            options=[
                                {'label': 'Pearson (corrélation linéaire)', 'value': 'pearson'},
                                {'label': 'Spearman (corrélation monotone)', 'value': 'spearman'}
                            ],
                            value='pearson'
                        )
                    ], width=12)
                ]),
                dbc.Button("Lancer le test", 
                         id="btn-run-correlation", 
                         color="success",
                         className="mt-3"),
                html.Div(id="correlation-results", className="mt-4")
            ])
        ])
    
    # Test Chi-carré
    elif triggered_id == "btn-chi-squared":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Test d'Indépendance du Chi-carré", className="text-white"),
                html.P("Vérifie s'il existe une association entre deux variables catégoriques. Le test compare les "
                      "fréquences observées avec les fréquences attendues sous l'hypothèse d'indépendance. "
                      "Utile pour analyser les tables de contingence.", className="text-dark mb-0")
            ], className="bg-info"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Variable catégorique 1 :"),
                        dcc.Dropdown(
                            id="chi-var1-select",
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=categorical_cols[0] if len(categorical_cols) > 0 else None
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Variable catégorique 2 :"),
                        dcc.Dropdown(
                            id="chi-var2-select",
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=categorical_cols[1] if len(categorical_cols) > 1 else None
                        )
                    ], width=6)
                ]),
                dbc.Button("Lancer le test", 
                         id="btn-run-chi", 
                         color="success",
                         className="mt-3"),
                html.Div(id="chi-results", className="mt-4")
            ])
        ])
    
    # Test t de Student
    elif triggered_id == "btn-t-student":
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Test t de Student", className="text-white"),
                html.P("Compare les moyennes de deux groupes indépendants. Suppose que les données sont normalement "
                      "distribuées et que les variances sont égales (test de Welch disponible si variances inégales). "
                      "Idéal pour comparer un résultat numérique entre deux catégories.", className="text-dark mb-0")
            ], className="bg-info"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Variable numérique :"),
                        dcc.Dropdown(
                            id="t-num-var-select",
                            options=[{'label': col, 'value': col} for col in numerical_cols],
                            value=numerical_cols[0] if numerical_cols else None
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Variable catégorique (2 groupes) :"),
                        dcc.Dropdown(
                            id="t-cat-var-select",
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=categorical_cols[0] if categorical_cols else None
                        )
                    ], width=6)
                ]),
                dbc.Button("Lancer le test", 
                         id="btn-run-t", 
                         color="success",
                         className="mt-3"),
                html.Div(id="t-results", className="mt-4")
            ])
        ])
    
    return html.Div()

# Callbacks pour exécuter les tests
@app.callback(
    Output("normality-results", "children", allow_duplicate=True),
    [Input("btn-run-normality", "n_clicks")],
    [State("normality-var-select", "value"),
     State("normality-test-type", "value"),
     State("store-data", "data")],
    prevent_initial_call=True
)
def run_normality_test(n_clicks, var, test_type, data):
    if n_clicks is None:
        raise PreventUpdate
    
    df = pd.DataFrame(data)
    results = []
    
    try:
        if test_type == 'shapiro':
            stat, p = shapiro(df[var].dropna())
            test_name = "Shapiro-Wilk"
        elif test_type == 'ks':
            # Standardize data for better KS test results
            data_clean = df[var].dropna()
            data_normalized = (data_clean - data_clean.mean()) / data_clean.std()
            stat, p = kstest(data_normalized, 'norm')
            test_name = "Kolmogorov-Smirnov"
        else:
            raise ValueError("Type de test non reconnu")
            
        conclusion = "Distribution normale" if p > 0.05 else "Distribution non normale"
        
        results = [
            html.H5(f"Résultats du test {test_name} :"),
            dbc.Row([
                dbc.Col(html.Strong("Statistique de test :"), width=4),
                dbc.Col(f"{stat:.4f}", width=8)
            ]),
            dbc.Row([
                dbc.Col(html.Strong("Valeur p :"), width=4),
                dbc.Col(f"{p:.4f}", width=8)
            ]),
            dbc.Alert(
                conclusion,
                color="success" if p > 0.05 else "danger",
                className="mt-3"
            )
        ]
        
    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")
    
    return results

@app.callback(
    Output("correlation-results", "children"),
    [Input("btn-run-correlation", "n_clicks")],
    [State("corr-var1-select", "value"),
     State("corr-var2-select", "value"),
     State("corr-method-select", "value"),
     State("store-data", "data")]
)
def run_correlation_test(n_clicks, var1, var2, method, data):
    if n_clicks is None:
        raise PreventUpdate
    
    df = pd.DataFrame(data).dropna()
    results = []
    
    try:
        if method == 'pearson':
            corr, p = pearsonr(df[var1], df[var2])
        else:
            corr, p = spearmanr(df[var1], df[var2])
            
        results.append(html.H5(f"Corrélation de {method.capitalize()} :"))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Coefficient :"), width=4),
            dbc.Col(f"{corr:.3f}", width=8)
        ]))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Valeur p :"), width=4),
            dbc.Col(f"{p:.4f}", width=8)
        ]))
        results.append(dbc.Alert(
            "Corrélation significative" if p < 0.05 else "Pas de corrélation significative",
            color="success" if p < 0.05 else "warning",
            className="mt-3"
        ))
        
    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")
    
    return results

@app.callback(
    Output("chi-results", "children"),
    [Input("btn-run-chi", "n_clicks")],
    [State("chi-var1-select", "value"),
     State("chi-var2-select", "value"),
     State("store-data", "data")]
)
def run_chi_test(n_clicks, var1, var2, data):
    if n_clicks is None:
        raise PreventUpdate
    
    df = pd.DataFrame(data)
    results = []
    
    try:
        contingency = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        results.append(html.H5("Résultats du test du Chi-carré :"))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Statistique Chi2 :"), width=4),
            dbc.Col(f"{chi2:.2f}", width=8)
        ]))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Valeur p :"), width=4),
            dbc.Col(f"{p:.4f}", width=8)
        ]))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Degrés de liberté :"), width=4),
            dbc.Col(dof, width=8)
        ]))
        results.append(dbc.Alert(
            "Dépendance significative" if p < 0.05 else "Indépendance des variables",
            color="success" if p < 0.05 else "warning",
            className="mt-3"
        ))
        
        # Tableau de contingence
        results.append(html.H5("Tableau de contingence :", className="mt-4"))
        results.append(dash_table.DataTable(
            columns=[{"name": str(i), "id": str(i)} for i in contingency.columns],
            data=contingency.reset_index().to_dict('records'),
            page_size=5
        ))
        
    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")
    
    return results

@app.callback(
    Output("t-results", "children"),
    [Input("btn-run-t", "n_clicks")],
    [State("t-num-var-select", "value"),
     State("t-cat-var-select", "value"),
     State("store-data", "data")]
)
def run_t_test(n_clicks, num_var, cat_var, data):
    if n_clicks is None:
        raise PreventUpdate
    
    df = pd.DataFrame(data).dropna()
    results = []
    
    try:
        groups = df[cat_var].unique()
        if len(groups) != 2:
            return dbc.Alert("La variable catégorique doit avoir exactement 2 groupes", color="danger")
            
        group1 = df[df[cat_var] == groups[0]][num_var]
        group2 = df[df[cat_var] == groups[1]][num_var]
        
        t_stat, p = ttest_ind(group1, group2)
        
        results.append(html.H5("Résultats du test t de Student :"))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Statistique t :"), width=4),
            dbc.Col(f"{t_stat:.3f}", width=8)
        ]))
        results.append(dbc.Row([
            dbc.Col(html.Strong("Valeur p :"), width=4),
            dbc.Col(f"{p:.4f}", width=8)
        ]))
        results.append(dbc.Alert(
            "Différence significative" if p < 0.05 else "Pas de différence significative",
            color="success" if p < 0.05 else "warning",
            className="mt-3"
        ))
        
        # Boxplot comparatif
        fig = px.box(df, x=cat_var, y=num_var, color=cat_var)
        results.append(dcc.Graph(figure=fig))
        
    except Exception as e:
        return dbc.Alert(f"Erreur : {str(e)}", color="danger")
    
    return results

# =============================================
#partie bloc_notes
# =============================================

@app.callback(
    [Output('download-txt', 'data'),
     Output('notes-status', 'children', allow_duplicate=True)],
    [Input('export-txt-btn', 'n_clicks')],
    [State('notes-textarea', 'value'),
     State('export-filename', 'value')],
    prevent_initial_call=True
)

def export_to_txt(n_clicks, notes_content, filename):
    if n_clicks is None or not notes_content:
        raise PreventUpdate

    try:
        # Nettoyage du nom de fichier
        clean_name = "notes_analyse"
        if filename:
            clean_name = re.sub(r'[^\w-]', '', filename)[:40]
            if not clean_name:
                clean_name = "notes_analyse"

        # Génération du contenu texte
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
        header = f"Vos Notes d'Analyse\nGénéré le: {timestamp}\n\n"
        full_text = header + notes_content

        # Version CORRIGEE avec les clés valides
        return (
            dict(content=full_text, filename=f"{clean_name}.txt", type="text/plain"),
            f"Fichier texte '{clean_name}.txt' généré avec succès !"
        )

    except Exception as e:
        return (
            None,
            f"Erreur lors de la génération du fichier TXT : {str(e)}"
        )
    
# =============================================
# Initialisation de l'Application
# =============================================
if __name__ == '__main__':  
    app.run(debug=True)


# Add a callback to capture changes in the conversion data store
@app.callback(
    Output('conversion-data-store', 'data', allow_duplicate=True),
    [Input('conversion-data-store', 'modified_timestamp')],
    prevent_initial_call=True
)
def update_conversion_data(timestamp):
    if not timestamp:
        raise PreventUpdate
    # On ne fait rien ici, car nous voulons juste réagir aux changements
    # dans le store de données de conversion
    return dash.no_update

