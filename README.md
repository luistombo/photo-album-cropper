# photo-album-cropper

CLI en Python para procesar fotos digitales tomadas a albumes o fotos impresas de papel. Detecta la foto dentro de la imagen, corrige perspectiva/keystone, recorta, orienta la escena cuando puede y aplica una restauracion suave de brillo, contraste, color y reflejos.

El proyecto combina dos enfoques:

- OpenCV local para deteccion, recorte, correccion de perspectiva y mejora de imagen.
- Gemini opcional para detectar esquinas y orientacion semantica en casos dificiles: fotos giradas, personas de costado, horizontes, edificios, texto o imagenes antiguas con bordes poco claros.

No hace super-resolution ni intenta inventar detalles. La salida busca mejorar fotos de papel antiguas sin cambiar la identidad visual de la imagen original.

## Requisitos

- Windows 10/11.
- Python 3.11 o superior.
- Conexion a internet solo si se usa Gemini.
- Una API key de Gemini solo si queres usar la deteccion avanzada por IA.

Dependencias Python:

- `opencv-python`
- `Pillow`
- `numpy`
- `tqdm`

Todas estan listadas en `requirements.txt`.

## Instalacion Rapida En Windows

Desde PowerShell o CMD, parado en la carpeta del proyecto:

```bat
install.bat
```

Ese script:

1. Crea el entorno virtual `.venv` si no existe.
2. Actualiza `pip`.
3. Instala las dependencias de `requirements.txt`.
4. Crea las carpetas `input/`, `output/cropped/`, `output/debug/` y `output/needs_review/`.

Si preferis hacerlo manualmente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No hace falta activar el entorno virtual. Los comandos usan directamente `.venv\Scripts\python.exe`, lo que evita problemas con la politica de ejecucion de scripts de PowerShell.

## Uso Rapido

1. Copia tus imagenes originales dentro de `input/`.
2. Ejecuta:

```bat
run.bat
```

El script ejecuta:

```powershell
.\.venv\Scripts\python.exe main.py --input input --output output
```

Tambien podes pasar argumentos extra:

```bat
run.bat --config config.json
```

O usar el CLI directamente:

```powershell
.\.venv\Scripts\python.exe main.py --input input --output output --config config.json
```

Formatos soportados de entrada: JPG, JPEG, PNG y WEBP.

## Estructura De Carpetas

```text
photo-album-cropper/
  input/
    foto_001.jpg
    foto_002.webp
  output/
    cropped/
    debug/
    needs_review/
```

- `input/`: fotos originales. El programa nunca modifica estos archivos.
- `output/cropped/`: fotos recortadas, enderezadas y corregidas.
- `output/debug/`: imagenes con el contorno detectado dibujado encima.
- `output/needs_review/`: originales que no tuvieron una deteccion confiable o fallaron.

Las fotos y salidas estan ignoradas por Git para evitar subir material privado al repositorio.

## Gemini Opcional

Gemini mejora mucho los casos donde OpenCV solo no alcanza:

- Fotos tomadas rotadas 90, 180 o 270 grados.
- Fotos de papel con bordes poco visibles.
- Keystone fuerte por haber fotografiado desde un angulo.
- Paisajes o personas donde hace falta entender cual es la orientacion correcta.
- Albumes con margen, plastico, separadores o contenido alrededor de la foto real.

El programa usa Gemini solo para estimar esquinas y orientacion. El recorte, warp y mejora de color se hacen localmente.

### Que API Hay Que Habilitar

Necesitas una API key de la **Gemini API** para desarrolladores. La forma recomendada es crearla desde Google AI Studio:

- Google AI Studio API keys: https://aistudio.google.com/app/apikey
- Guia oficial de API keys: https://ai.google.dev/gemini-api/docs/api-key
- Billing y tiers: https://ai.google.dev/gemini-api/docs/billing
- Rate limits/cuotas: https://ai.google.dev/gemini-api/docs/rate-limits

En Google AI Studio, crea o selecciona un proyecto y genera una API key. Para produccion o uso intensivo, vincula billing al proyecto. Segun la documentacion oficial, las cuotas dependen del proyecto, modelo y tier de uso; al pasar a tier pago suben los limites y pueden aplicar limites por requests por minuto, tokens por minuto, requests por dia y gasto.

Si vas a restringir la key, la documentacion actual recomienda restringirla a la Gemini API. No uses una key irrestricta si la vas a compartir o guardar en una maquina que no controlas.

### Guardar La API Key En Windows

Opcion recomendada en PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "TU_API_KEY", "User")
```

Cierra y vuelve a abrir PowerShell/CMD para que la variable aparezca en nuevas terminales.

Alternativa desde CMD o PowerShell:

```bat
setx GEMINI_API_KEY "TU_API_KEY"
```

Para probar si quedo disponible en la terminal actual:

```powershell
$env:GEMINI_API_KEY
```

Importante: no pegues tu API key dentro del README ni la commitees en Git. Este programa la lee desde la variable de entorno `GEMINI_API_KEY` o desde el registro de entorno de usuario de Windows.

### Activar Gemini En config.json

Edita `config.json`:

```json
{
  "gemini": {
    "enabled": true,
    "mode": "always",
    "model": "gemini-3.5-flash",
    "min_confidence": 0.55,
    "max_retries": 2,
    "retry_delay_seconds": 2.0,
    "request_timeout_seconds": 25.0,
    "orientation_check": true,
    "orientation_min_confidence": 0.65,
    "orientation_max_retries": 1,
    "fallback_methods": ["album_edges", "foreground_bbox", "full_frame", "not_found", "unreasonable"]
  }
}
```

Modos:

- `fallback`: usa Gemini solo si la deteccion local fue incierta.
- `always`: consulta Gemini siempre. Es mejor para tandas complicadas, pero consume mas cuota.

`orientation_check` hace una segunda consulta liviana despues del recorte para corregir si la foto quedo de lado o invertida.

## Configuracion Principal

`config.json` controla el comportamiento del procesador:

```json
{
  "margin_percent": 1.0,
  "debug": true,
  "color_correction_strength": 0.6,
  "keep_white_border": false,
  "accept_already_cropped": true,
  "rotate_to_landscape": false,
  "output_quality": 95,
  "gamma": 1.0,
  "saturation": 1.05,
  "restoration": {
    "enabled": true,
    "glare_reduction_strength": 0.35,
    "white_balance_strength": 0.45,
    "age_cast_reduction_strength": 0.55,
    "local_contrast_strength": 0.45,
    "shadow_recovery_strength": 0.28,
    "vibrance": 1.12
  },
  "gemini": {
    "enabled": false,
    "mode": "fallback",
    "model": "gemini-3.5-flash",
    "min_confidence": 0.55,
    "max_retries": 2,
    "retry_delay_seconds": 2.0,
    "request_timeout_seconds": 25.0,
    "orientation_check": true,
    "orientation_min_confidence": 0.65,
    "orientation_max_retries": 1,
    "fallback_methods": ["album_edges", "foreground_bbox", "full_frame", "not_found", "unreasonable"]
  }
}
```

Opciones generales:

- `margin_percent`: margen extra aplicado al cuadrilatero detectado.
- `debug`: genera imagenes con el contorno detectado.
- `color_correction_strength`: intensidad general de las correcciones clasicas, de `0.0` a `1.0`.
- `keep_white_border`: conserva mas borde blanco de la foto impresa.
- `accept_already_cropped`: acepta la imagen completa cuando ya parece pre-recortada.
- `rotate_to_landscape`: fuerza recortes verticales a horizontal si corresponde.
- `output_quality`: calidad JPG de salida.
- `gamma`: ajuste leve de luminosidad. Valores tipicos: `0.95` a `1.08`.
- `saturation`: ajuste leve de saturacion. Valores tipicos: `1.0` a `1.1`.

Restauracion:

- `restoration.enabled`: activa restauracion local suave.
- `restoration.glare_reduction_strength`: reduce reflejos especulares pequenos o medianos.
- `restoration.white_balance_strength`: compensa dominantes por luz ambiente o papel envejecido.
- `restoration.age_cast_reduction_strength`: reduce amarilleo, magenta o cian en medios tonos.
- `restoration.local_contrast_strength`: recupera contraste local sin efecto HDR agresivo.
- `restoration.shadow_recovery_strength`: levanta sombras profundas moderadamente.
- `restoration.vibrance`: recupera color en zonas apagadas sin sobresaturar zonas ya intensas.

## Como Funciona

1. Lee la imagen con Pillow y respeta orientacion EXIF.
2. Usa OpenCV para buscar bordes en gris y LAB.
3. Aplica Canny, operaciones morfologicas y busqueda de contornos.
4. Elige un cuadrilatero razonable para la foto impresa.
5. Si Gemini esta activo, puede reemplazar la deteccion local con esquinas semanticas.
6. Aplica `cv2.getPerspectiveTransform` para corregir perspectiva.
7. Hace una revision opcional de orientacion con Gemini despues del recorte.
8. Aplica correccion de color, balance de blancos, contraste local, sombras y reduccion de reflejos.
9. Guarda JPG final en `output/cropped/`.

Cuando Gemini esta activo, las cuatro esquinas se piden en orden semantico final: arriba-izquierda, arriba-derecha, abajo-derecha y abajo-izquierda de la foto cuando la escena esta derecha. Eso permite corregir fotos tomadas de costado o invertidas durante el warp de perspectiva.

## Problemas Frecuentes

### PowerShell No Deja Activar .venv

No necesitas activar el entorno. Usa:

```powershell
.\.venv\Scripts\python.exe main.py --input input --output output
```

O directamente:

```bat
run.bat
```

### Gemini Devuelve 429 RESOURCE_EXHAUSTED

Es un limite de cuota o rate limit. Revisa:

- Que el proyecto tenga billing activo si necesitas tier pago.
- Que estes usando la API key del mismo proyecto donde habilitaste Gemini/billing.
- Los limites del modelo elegido en Google AI Studio.
- La pagina de rate limits de la documentacion oficial.

Para gastar menos cuota, usa `"mode": "fallback"` en vez de `"always"`.

### La Deteccion No Es Buena

Proba estas opciones:

- Activa Gemini con `"enabled": true`.
- Usa `"mode": "always"` para tandas complejas.
- Aumenta levemente `margin_percent`, por ejemplo `1.5`.
- Deja `debug` en `true` y revisa `output/debug/`.
- Si ya recortaste manualmente, usa `"accept_already_cropped": true`.

## Limitaciones

La deteccion automatica depende de que exista informacion visual suficiente. Fondos muy parecidos al papel, reflejos grandes, sombras duras, manos tapando esquinas, fotos parcialmente fuera de cuadro o varias fotos superpuestas pueden requerir revision manual.
