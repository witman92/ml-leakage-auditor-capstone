% ========================================
% CLASE PERSONALIZADA PARA CAPSTONE PROJECT
% Universidad de Las Américas
%
% Autor: Christian Pieringer
% Fecha: 14/10/2025
%
% Version: 2.0
% Last update: 5/09/2026
% ========================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{udla-capstone}[2025/10/14 v1.0 Clase para Capstone Project UDLA]

% Opciones por defecto
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

% Cargar la clase base
\LoadClass[11pt,letterpaper]{report}

% ========================================
% PAQUETES BÁSICOS
% ========================================
\RequirePackage[utf8]{inputenc}
\RequirePackage[spanish]{babel}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{subcaption}        % Subfiguras
\RequirePackage{float}             % Mejor control de posicionamiento
\RequirePackage{caption}           % Personalización de leyendas
\RequirePackage{booktabs}          % Tablas profesionales
\RequirePackage{array}             % Columnas de tabla avanzadas
\RequirePackage{multirow}          % Celdas que abarcan múltiples filas
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage{setspace}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}

% Configuración de biblatex para formato académico
\RequirePackage{csquotes}
% style=apa: citas y referencias en formato APA 7 (exigido por la rúbrica)
\RequirePackage[style=apa,backend=biber]{biblatex}

% ========================================
% CONFIGURACIÓN DE MÁRGENES
% ========================================
\geometry{
    left=2.5cm,
    right=2cm,
    top=2.5cm,
    bottom=2.5cm
}

% ========================================
% CONFIGURACIÓN DE ESPACIADO
% ========================================
\onehalfspacing

% ========================================
% CONFIGURACIÓN DE FUENTE
% ========================================
\renewcommand{\familydefault}{\sfdefault}
\renewcommand{\rmdefault}{\sfdefault}

% ========================================
% CONFIGURACIÓN DE ENCABEZADOS Y PIES DE PÁGINA
% ========================================
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Asegurar que el estilo plain también use la numeración a la derecha
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[R]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

% ========================================
% CONFIGURACIÓN DE TÍTULOS
% ========================================
\titleformat{\chapter}[block]{\Large\bfseries}{\thechapter.}{1em}{\Large}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{}

% Configuración del título del índice (solo para el TOC)
\titleformat{\section}[block]{\large\bfseries}{\thesection}{1em}{}

% ========================================
% CONFIGURACIÓN DE ENTORNOS PERSONALIZADOS
% ========================================

% Configuración del abstract para que comience en el margen superior
\renewenvironment{abstract}
{%\small
\begin{center}
\bfseries ABSTRACT
\end{center}
\vspace{0.5cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}}
{\par\vspace{0.5cm}}

% Configuración de resumen en español
\newenvironment{resumen}
{%\small
\begin{center}
\bfseries RESUMEN
\end{center}
\vspace{0.5cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}}
{\par\vspace{0.5cm}}

% Configuración de agradecimientos
\newenvironment{acknowledgments}
{\small
\begin{center}
\bfseries AGRADECIMIENTOS
\end{center}
\vspace{0.5cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}}
{\par\vspace{0.5cm}}

% ========================================
% CONFIGURACIÓN DE LISTAS
% ========================================
\setlist[itemize]{
    itemsep=0.23em,
    parsep=0pt,
    topsep=0.4em,
    partopsep=0pt
}


% ========================================
% CONFIGURACIÓN DE LISTAS NUMERADAS
% ========================================
\setlist[enumerate]{
    itemsep=0.25em,
    parsep=0pt,
    topsep=0.4em,
    partopsep=0pt
}

% ========================================
% CONFIGURACIÓN DE HIPERVÍNCULOS
% ========================================
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    filecolor=black,      
    urlcolor=black,
    citecolor=black
}

% ========================================
% CONFIGURACIÓN DE NOMBRES
% ========================================
% Configuración del nombre de la tabla de contenidos
\renewcommand{\contentsname}{ÍNDICE GENERAL}

% Configuración del nombre de las tablas y figuras (después de babel)
\AtBeginDocument{%
    \renewcommand{\tablename}{Tabla}%
    \renewcommand{\figurename}{Figura}%
}

% ========================================
% CONFIGURACIÓN AVANZADA DE FIGURAS
% ========================================
% Configuración de caption para mejor formato (APA 7ª edición)
\captionsetup{
    font=small,
    labelfont=bf,
    textfont=it,
    justification=raggedright,
    singlelinecheck=false,
    width=0.90\textwidth,
    labelsep=period
}

% Configuración específica para subfiguras (APA 7ª edición)
\captionsetup[subfigure]{
    font=small,
    labelfont=bf,
    textfont=it,
    justification=centering,
    singlelinecheck=true,
    width=\textwidth
}

% Configuración de espaciado para figuras
\setlength{\intextsep}{12pt plus 2pt minus 2pt}

% Configuración de float para mejor posicionamiento
\floatplacement{figure}{H}
\floatplacement{table}{H}

% ========================================
% COMANDOS PERSONALIZADOS
% ========================================

% Contador para anexos
\newcounter{anexo}
\setcounter{anexo}{0}
\renewcommand{\theanexo}{\Alph{anexo}}   % \ref{} a un anexo imprime su letra (A, B, ...)

% Comando personalizado para anexos con numeración automática
\newcommand{\anexo}[1]{%
    \refstepcounter{anexo}%                % refstep: habilita \label/\ref al anexo
    \chapter*{ANEXO \Alph{anexo}. #1}%
    \addcontentsline{toc}{chapter}{ANEXO \Alph{anexo}. #1}%
}

% ========================================
% COMANDOS PARA INFORMACIÓN DEL PROYECTO
% ========================================

% Definir comandos internos primero
\newcommand{\@proyectotitulo}{Título del Proyecto de Capstone}
\newcommand{\@proyectoautor}{Nombre Completo del Estudiante}
\newcommand{\@proyectoprofesor}{Nombre del Profesor Guía}
\newcommand{\@proyectocoprofesor}{Nombre del Profesor Co-guía}
\newcommand{\@proyectofecha}{Octubre, 2025}
\newcommand{\@proyectociudad}{Santiago de Chile}

% Comandos para definir información del proyecto
\newcommand{\proyectotitulo}[1]{\renewcommand{\@proyectotitulo}{#1}}
\newcommand{\proyectoautor}[1]{\renewcommand{\@proyectoautor}{#1}}
\newcommand{\proyectoprofesor}[1]{\renewcommand{\@proyectoprofesor}{#1}}
\newcommand{\proyectocoprofesor}[1]{\renewcommand{\@proyectocoprofesor}{#1}}
\newcommand{\proyectofecha}[1]{\renewcommand{\@proyectofecha}{#1}}
\newcommand{\proyectociudad}[1]{\renewcommand{\@proyectociudad}{#1}}

% Comandos para usar en el documento
\newcommand{\tituloproyecto}{\@proyectotitulo}
\newcommand{\autorproyecto}{\@proyectoautor}
\newcommand{\profesorproyecto}{\@proyectoprofesor}
\newcommand{\coprofesorproyecto}{\@proyectocoprofesor}
\newcommand{\fechaproyecto}{\@proyectofecha}
\newcommand{\ciudadproyecto}{\@proyectociudad}

% Comando para la página de título completa
\newcommand{\portada}{%
    \begin{titlepage}
        \thispagestyle{empty} % Sin número de página
        
        \begin{center}
            % Logo UDLA (ajustar ruta según ubicación)
            \includegraphics[width=5.5cm,height=1.5cm]{assets/UDLA_logo.png}

            \vspace*{4cm}
            
            % Título del proyecto (centrado, negrita, máximo 12 palabras)
            % Aumentar separación entre líneas del título
            \begin{spacing}{1.7}
            {\huge\textbf{\uppercase{\tituloproyecto}}}
            \end{spacing}
            
            \vspace{2cm}
            
            % Información del autor
            {\Large\textbf{\autorproyecto}}
            
            \vspace{0.5cm}
            
            % Afiliación institucional
            {\Large
            Universidad de Las Américas\\
            Facultad de Ingeniería y Negocios\\
            Magíster en Ciencia de Datos
            }
            \vspace{2.5cm}
            
            % Información del curso
            \large\textbf{Capstone Project}\\
            \large\textbf{Profesor Guía:} \profesorproyecto\\
            \large\textbf{Profesor Co-guía:} \coprofesorproyecto
            
            % \vspace{2cm}
            \vfill
            
            % Fecha
            \large\textbf{\ciudadproyecto, \fechaproyecto}
            
        \end{center}
        
    \end{titlepage}
}

% ========================================
% CONFIGURACIÓN FINAL
% ========================================
\endinput
