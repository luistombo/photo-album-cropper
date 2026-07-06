# photo-album-cropper

Proyecto Windows en Python para procesar fotos digitales tomadas a albumes de fotos de papel. Detecta la foto impresa dentro de la imagen, corrige perspectiva, recorta, rota si corresponde y aplica correcciones clasicas suaves de color, brillo y contraste.

No usa super-resolution ni altera facciones o inventa detalles. Opcionalmente puede usar Gemini para estimar las esquinas de la foto cuando la deteccion clasica no es confiable; el recorte y la correccion se hacen localmente.

## Requisitos

- Python 3.11+
- OpenCV
- Pillow
- numpy
- tqdm

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura esperada

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

La carpeta `input/` contiene las fotos originales. El programa nunca modifica esos archivos.

## Uso

```powershell
python main.py --input input --output output
```

Archivos soportados: JPG, JPEG, PNG y WEBP.

## Salidas

- `output/cropped/`: fotos recortadas, corregidas y exportadas como JPG.
- `output/debug/`: imagenes con el contorno detectado dibujado encima.
- `output/needs_review/`: originales que no tuvieron una deteccion confiable o fallaron durante el proceso.

Si el programa no detecta bien una imagen, no inventa un recorte: copia el original a `needs_review/`.

## Configuracion

Editar `config.json`:

```json
{
  "margin_percent": 1.0,
  "debug": true,
  "color_correction_strength": 0.6,
  "keep_white_border": false,
  "accept_already_cropped": false,
  "rotate_to_landscape": true,
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
    "fallback_methods": ["album_edges", "foreground_bbox", "full_frame", "not_found", "unreasonable"]
  }
}
```

- `margin_percent`: margen extra aplicado al cuadrilatero detectado.
- `debug`: crea imagenes con el contorno detectado.
- `color_correction_strength`: intensidad general de correcciones clasicas, de `0.0` a `1.0`.
- `keep_white_border`: conserva mas del borde blanco de la foto impresa cuando esta en `true`.
- `accept_already_cropped`: acepta la imagen completa como salida cuando ya fue pre-recortada a mano.
- `rotate_to_landscape`: rota recortes verticales a horizontal cuando esta en `true`.
- `output_quality`: calidad de salida JPG/WEBP.
- `gamma`: ajuste leve de luminosidad. Valores tipicos: `0.95` a `1.08`.
- `saturation`: ajuste leve de saturacion. Valores tipicos: `1.0` a `1.1`.
- `restoration.enabled`: activa restauracion local suave para fotos de papel antiguas.
- `restoration.glare_reduction_strength`: atenua reflejos especulares pequenos o medianos detectados por brillo local.
- `restoration.white_balance_strength`: compensa dominantes de color por envejecimiento o luz ambiente.
- `restoration.age_cast_reduction_strength`: reduce amarilleo/magenta/cian en medios tonos.
- `restoration.local_contrast_strength`: recupera contraste local sin hacer un HDR agresivo.
- `restoration.shadow_recovery_strength`: levanta sombras profundas de forma moderada.
- `restoration.vibrance`: recupera color en zonas apagadas sin sobresaturar zonas ya intensas.
- `gemini.enabled`: usa Gemini como ayuda opcional para detectar esquinas. Requiere `GEMINI_API_KEY`.
- `gemini.mode`: `fallback` usa Gemini solo en metodos inciertos; `always` lo consulta siempre.
- `gemini.model`: modelo de Gemini usado para vision.
- `gemini.min_confidence`: confianza minima aceptada para reemplazar la deteccion local.
- `gemini.max_retries`: reintentos ante saturacion temporal o errores transitorios.
- `gemini.retry_delay_seconds`: espera base entre reintentos.
- `gemini.request_timeout_seconds`: tiempo maximo de espera por respuesta de Gemini antes de reintentar.
- `gemini.orientation_check`: hace una segunda consulta liviana despues del recorte para corregir fotos que siguen de lado o invertidas.
- `gemini.orientation_min_confidence`: confianza minima para aplicar esa rotacion post-recorte.
- `gemini.orientation_max_retries`: reintentos para la consulta de orientacion post-recorte.
- `gemini.fallback_methods`: metodos locales que disparan el fallback de Gemini.

Cuando Gemini esta activo, ordena las cuatro esquinas segun la orientacion semantica final: personas de pie, horizontes, edificios, autos, cielo/suelo o texto reconocible en su orientacion natural. Esto permite corregir fotos tomadas de costado o invertidas durante el mismo warp de perspectiva.

## Como funciona

1. Lee la imagen con Pillow y respeta la orientacion EXIF.
2. Usa OpenCV para buscar bordes en gris y LAB.
3. Aplica Canny, dilatacion/cierre morfologico y busqueda de contornos.
4. Elige el mayor cuadrilatero razonable.
5. Si no hay cuadrilatero, intenta un fallback con bounding box del area distinta al fondo.
6. Ordena las esquinas y aplica `cv2.getPerspectiveTransform`.
7. Aplica balance de blancos gray-world, autocontraste moderado en LAB, gamma configurable y saturacion leve.
8. Si `restoration.enabled` esta activo, reduce dominantes de color, recupera contraste local, levanta sombras y atenua reflejos especulares con una mascara conservadora.
9. Protege altas luces para evitar quemarlas.

## Limitaciones

La deteccion automatica depende de que exista contraste suficiente entre la foto impresa y el fondo. Fondos muy parecidos al papel, reflejos fuertes, sombras duras, manos tapando esquinas o varias fotos en una misma toma pueden requerir revision manual.
