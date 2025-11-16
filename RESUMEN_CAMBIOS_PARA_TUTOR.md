# Resumen de Cambios Realizados en la Memoria del TFM

**Fecha:** 16 de noviembre de 2025
**Estudiante:** Pablo
**Cambios realizados en respuesta a los comentarios del tutor**

---

## 1. Reducción Significativa de Longitud

### Capítulo 2 - Metodología
- **Antes:** 1,147 líneas
- **Después:** 778 líneas
- **Reducción:** 369 líneas (32% menos)

### Acciones específicas:
1. ✅ **Movido a Anexo A** (~210 líneas): Explicaciones técnicas detalladas sobre:
   - Vision Transformers (ViT) - arquitectura completa
   - Self-Attention mechanism - paso a paso
   - Multi-Head Attention - detalles matemáticos
   - CLIP - entrenamiento contrastivo completo

2. ✅ **Condensado SCLIP** (~150 líneas → ~40 líneas):
   - Eliminados Steps 1-8 detallados paso a paso
   - Reemplazado con resumen de 4 pasos esenciales
   - Mantenida solo la innovación clave (CSA)
   - Referencia al anexo para detalles

3. ✅ **Eliminado contenido duplicado** (~60 líneas):
   - Steps 6, 7, 8 que estaban repetidos
   - Explicaciones redundantes de template encoding

---

## 2. Marcadores Claros de Contribución

### Añadido marcador visual `[Our Contribution]` en:

**Contribución Principal:**
- ✅ **Intelligent Prompt Extraction** (Sección 2.X)
  - Marcado como: `[Main Contribution]`
  - 96% reducción de prompts (50-300 vs 4,096)
  - 68.09% mIoU en PASCAL VOC

**Contribuciones Secundarias:**
- ✅ **Descriptor Files** (Sección 2.X.1)
  - Marcado como: `[Our Contribution]`
  - Archivo cls_voc21.txt con términos múltiples
  - Background: 86.90% IoU

- ✅ **Template Optimization** (Sección 2.X.2)
  - Marcado como: `[Our Contribution]`
  - Evaluación de 5 estrategias
  - ImageNet-80 seleccionado para máxima precisión

- ✅ **Computational Optimizations** (Sección 2.X.3)
  - Marcado como: `[Our Contribution]`
  - FP16, torch.compile, batched prompting
  - 3-4× speedup acumulativo

**Trabajo Previo Adoptado:**
- ✅ **SCLIP** - Claramente marcado como trabajo previo \cite{sclip2024}
  - Nota explícita: "SCLIP is prior work that we adopt"
  - Nuestra contribución: usar SCLIP para prompting inteligente

---

## 3. Organización del Contenido

### Nuevo Anexo A: Fundamentos Técnicos
- **Archivo creado:** `AnexoA_Fundamentos.tex`
- **Contenido:**
  - Vision Transformer (ViT) - arquitectura completa
  - Self-Attention mechanism - matemáticas detalladas
  - CLIP - contrastive learning objetivo
- **Propósito:** Mantener rigor técnico sin distraer del contenido principal

### Capítulo 2 Reorganizado:
```
2.1 Motivation and Design Philosophy
2.2 System Overview [con figura placeholder]
2.3 Technical Background [CONDENSADO - ref. Anexo A]
2.4 SCLIP Dense Prediction [CONDENSADO]
    2.4.1 Cross-Layer Self-Attention [núcleo]
    2.4.2 Dense Prediction Pipeline [4 pasos]
    2.4.3 Descriptor Files [OUR CONTRIBUTION]
    2.4.4 Template Strategies [OUR CONTRIBUTION]
    2.4.5 Computational Optimizations [OUR CONTRIBUTION]
2.5 Intelligent Prompt Extraction [MAIN CONTRIBUTION]
    2.5.1 Motivation
    2.5.2 Complete Pipeline [5 stages]
    2.5.3 Algorithm Summary
    2.5.4 Direct Class Assignment
2.6 Extension to Video [brevemente]
```

---

## 4. Enfoque en Contribuciones vs Estado del Arte

### Antes:
- Explicaciones extensas de métodos baseline (MaskCLIP, CLIPSeg, LSeg)
- Detalles exhaustivos de transformers (innecesario para TFM)
- Difícil distinguir qué es nuestro vs qué es trabajo previo

### Después:
- ✅ Baseline methods: solo tabla comparativa + 1-2 párrafos contexto
- ✅ Fundamentos técnicos: movidos a anexo
- ✅ Contribuciones propias: **marcadas visualmente** con recuadros
- ✅ Referencias claras: "SCLIP \cite{sclip2024} is prior work"

---

## 5. Resultados Actualizados (68.09% mIoU)

### Capítulo 3 - Experimentos:
- ✅ Tabla principal: 59.78% → **68.09%** mIoU
- ✅ Per-class IoU: actualizada con 21 clases completas
- ✅ Métricas comprehensivas: todas actualizadas
  - Precision: 68.28% → **81.13%**
  - Pixel Accuracy: 74.65% → **85.38%**
  - F1: 62.36% → **68.97%**
- ✅ Análisis actualizado: énfasis en background (86.90%) y descriptor files

### Capítulo 5 - Conclusiones:
- ✅ Actualizado: "surpassing ITACLIP (67.9%)"
- ✅ Gap cerrado: 30 pts → 21 pts vs métodos closed-vocabulary
- ✅ Énfasis en state-of-the-art training-free

---

## 6. Mejoras Documentales

### Añadido:
1. ✅ **Sección de Descriptor Files** (completa)
   - Motivación clara
   - Ejemplos de cls_voc21.txt
   - Implementación matemática
   - Impacto en performance

2. ✅ **Sección de Template Strategies** (completa)
   - 5 estrategias evaluadas
   - Tabla comparativa con speedups
   - Recomendaciones de uso

3. ✅ **Sección de Computational Optimizations** (completa)
   - FP16 mixed precision
   - torch.compile JIT
   - Batched prompting
   - Tabla de impacto acumulativo

### Mejorado:
- ✅ Figuras: añadidos placeholders con descripción de qué mostrar
- ✅ Ecuaciones: simplificadas donde posible
- ✅ Flujo narrativo: más conciso, menos divagación

---

## 7. Próximos Pasos Recomendados

### Prioridad Alta:
1. **Generar figuras** para los placeholders existentes:
   - Figure 2.1: CLIP-Guided Prompting Pipeline
   - Figure 3.X: Failure cases
   - Figure 3.X: Qualitative results

2. **Revisar Capítulo 3_SCLIP.tex**:
   - Parece duplicado con Capítulo 3.tex
   - Considerar fusionar o eliminar

3. **Acortar Introducción/Conclusión** si es necesario:
   - Aplicar mismo principio: concisión

### Prioridad Media:
4. **Revisar referencias bibliográficas**:
   - Asegurar que todas las citas están completas
   - Formato consistente

5. **Spell-check final**:
   - Revisar consistencia de términos
   - Inglés británico vs americano

---

## 8. Estadísticas Finales

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|---------|
| Líneas Cap. 2 | 1,147 | 778 | -32% |
| Anexos creados | 0 | 1 | +1 |
| Marcadores visuales | 0 | 5 | +5 |
| mIoU PASCAL VOC | 59.78% | **68.09%** | +8.31 pts |
| Claridad contribuciones | Baja | **Alta** | ✅ |

---

## Conclusión

Los cambios realizados abordan **todos los puntos** mencionados por el tutor:

1. ✅ **Longitud reducida** - 32% menos en Cap. 2, moviendo detalles técnicos a anexos
2. ✅ **Concisión** - eliminado relleno, mantenido solo esencial
3. ✅ **Claridad contribuciones** - marcadores visuales claros
4. ✅ **Enfoque correcto** - resaltado lo nuestro vs estado del arte
5. ✅ **Resultados actualizados** - 68.09% mIoU integrado
6. ✅ **Mejor estructura** - anexos para detalles, capítulo principal conciso

La memoria ahora es más **directa**, **clara** y **profesional**, mostrando claramente nuestras contribuciones sin abrumar con detalles técnicos innecesarios.
