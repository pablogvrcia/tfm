#!/usr/bin/env python3
"""
Script para comparar visualmente resultados de diferentes estrategias de segmentación.

Muestra para cada imagen:
- Imagen original
- Ground truth
- Mapa de confianzas CLIP
- Localización de centroides (por estrategia)
- Segmentaciones finales de cada configuración
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import torch


def load_results(result_dir):
    """Carga resultados de un directorio de benchmarks"""
    result_dir = Path(result_dir)

    # Buscar archivo de resultados JSON
    json_files = list(result_dir.glob("*_results.json"))
    if not json_files:
        raise FileNotFoundError(f"No se encontró archivo de resultados en {result_dir}")

    with open(json_files[0], 'r') as f:
        results = json.load(f)

    return results, result_dir


def load_image(image_path):
    """Carga imagen y convierte a array numpy"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def create_confidence_map(probs):
    """
    Crea visualización del mapa de confianzas.
    Muestra la confianza máxima en cada píxel.
    """
    max_conf = np.max(probs, axis=-1)
    return max_conf


def visualize_centroids(image_shape, prompts, class_names):
    """
    Crea visualización de los centroides seleccionados.

    Args:
        image_shape: (H, W, C) shape de la imagen
        prompts: lista de dicts con 'point', 'label', 'class_idx'
        class_names: lista de nombres de clases

    Returns:
        Imagen con puntos dibujados
    """
    H, W = image_shape[:2]
    vis = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Crear colormap para clases
    n_classes = len(class_names)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))[:, :3]
    colors = (colors * 255).astype(np.uint8)

    # Dibujar puntos
    for prompt in prompts:
        x, y = prompt['point']
        class_idx = prompt['class_idx']

        # Dibujar círculo
        color = tuple(int(c) for c in colors[class_idx % len(colors)])

        # Círculo más grande para visibilidad
        rr, cc = np.ogrid[-5:6, -5:6]
        mask = rr**2 + cc**2 <= 25  # radio 5 píxeles

        # Verificar límites
        y_coords = np.clip(y + rr[mask], 0, H - 1)
        x_coords = np.clip(x + cc[mask], 0, W - 1)

        vis[y_coords, x_coords] = color

    return vis


def create_class_legend(class_names, colormap='tab20'):
    """Crea leyenda de colores para las clases"""
    n_classes = len(class_names)
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_classes))

    patches = []
    for i, name in enumerate(class_names):
        patches.append(mpatches.Patch(color=colors[i % len(colors)], label=name))

    return patches


def compare_segmentations(image_idx, configs, dataset_name='coco-stuff', output_path=None):
    """
    Compara visualmente segmentaciones de diferentes configuraciones.

    Args:
        image_idx: índice de la imagen a visualizar
        configs: dict con nombre_config -> path_resultados
        dataset_name: nombre del dataset
        output_path: path para guardar visualización (opcional)
    """
    n_configs = len(configs)

    # Crear figura con grid
    # Fila 1: Original, GT, Confidence Map, Centroids
    # Fila 2+: Una fila por configuración
    fig = plt.figure(figsize=(20, 5 * (n_configs + 1)))

    # Cargar datos de la primera configuración para obtener imagen y GT
    first_config = list(configs.values())[0]
    results, result_dir = load_results(first_config)

    # Buscar visualizaciones guardadas
    vis_dir = result_dir / 'visualizations'
    if not vis_dir.exists():
        print(f"Warning: No se encontró directorio de visualizaciones en {result_dir}")
        return

    # Buscar archivos de la imagen
    vis_files = sorted(vis_dir.glob(f"sample_{image_idx:04d}_*.png"))
    if not vis_files:
        print(f"Warning: No se encontró visualización para sample {image_idx}")
        return

    # Cargar imagen original (asumiendo que está guardada)
    original_file = None
    gt_file = None
    pred_file = None

    for f in vis_files:
        if 'original' in f.name or 'image' in f.name:
            original_file = f
        elif 'gt' in f.name or 'ground_truth' in f.name:
            gt_file = f
        elif 'pred' in f.name or 'prediction' in f.name:
            pred_file = f

    # Fila 1: Visualizaciones comunes
    row = 0
    col = 0

    # Original
    if original_file:
        ax = plt.subplot(n_configs + 1, 4, row * 4 + col + 1)
        img = load_image(original_file)
        ax.imshow(img)
        ax.set_title('Imagen Original', fontsize=12, fontweight='bold')
        ax.axis('off')
        col += 1

    # Ground Truth
    if gt_file:
        ax = plt.subplot(n_configs + 1, 4, row * 4 + col + 1)
        gt = load_image(gt_file)
        ax.imshow(gt)
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')
        col += 1

    # Espacio para mapa de confianzas (si disponible)
    ax = plt.subplot(n_configs + 1, 4, row * 4 + col + 1)
    ax.text(0.5, 0.5, 'Confidence Map\n(requiere probs.npy)',
            ha='center', va='center', fontsize=10)
    ax.set_title('CLIP Confidence Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    col += 1

    # Espacio para centroides (si disponible)
    ax = plt.subplot(n_configs + 1, 4, row * 4 + col + 1)
    ax.text(0.5, 0.5, 'Centroids\n(requiere prompts.json)',
            ha='center', va='center', fontsize=10)
    ax.set_title('Prompt Centroids', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Filas 2+: Una por configuración
    for config_idx, (config_name, config_path) in enumerate(configs.items()):
        row = config_idx + 1
        results, result_dir = load_results(config_path)

        # Buscar visualización de predicción
        vis_dir = result_dir / 'visualizations'
        vis_files = sorted(vis_dir.glob(f"sample_{image_idx:04d}_*.png"))

        pred_file = None
        for f in vis_files:
            if 'pred' in f.name or 'prediction' in f.name:
                pred_file = f
                break

        # Mostrar predicción
        if pred_file:
            ax = plt.subplot(n_configs + 1, 4, row * 4 + 1)
            pred = load_image(pred_file)
            ax.imshow(pred)
            ax.set_title(f'{config_name}', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Mostrar métricas si están disponibles
            ax = plt.subplot(n_configs + 1, 4, row * 4 + 2)
            metrics_text = f"mIoU: {results.get('miou', 0):.2%}\n"
            metrics_text += f"Pixel Acc: {results.get('pixel_accuracy', 0):.2%}\n"
            metrics_text += f"F1: {results.get('f1', 0):.2%}"

            ax.text(0.1, 0.5, metrics_text, fontsize=11,
                   verticalalignment='center', family='monospace')
            ax.set_title('Métricas', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Mostrar configuración
            ax = plt.subplot(n_configs + 1, 4, row * 4 + 3)
            args = results.get('args', {})
            config_text = ""

            if 'improved_strategy' in args:
                config_text += f"Strategy: {args['improved_strategy']}\n"
            if 'template_strategy' in args:
                config_text += f"Templates: {args['template_strategy']}\n"
            if 'use_hybrid_voting' in args:
                hybrid = "Yes" if args['use_hybrid_voting'] else "No"
                config_text += f"Hybrid Voting: {hybrid}\n"
            if 'descriptor_file' in args and args['descriptor_file']:
                desc_file = Path(args['descriptor_file']).name
                config_text += f"Descriptors: {desc_file}\n"
            if 'confidence_weighted_centroid' in args and args['confidence_weighted_centroid']:
                config_text += f"Confidence Centroid: Yes\n"

            ax.text(0.1, 0.5, config_text, fontsize=10,
                   verticalalignment='center', family='monospace')
            ax.set_title('Configuración', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Tiempo de ejecución
            ax = plt.subplot(n_configs + 1, 4, row * 4 + 4)
            profiling = results.get('profiling', {})
            summary = profiling.get('summary', {})

            time_text = ""
            if 'total_inference' in summary:
                mean_time = summary['total_inference'].get('mean_time', 0)
                time_text += f"Mean Time: {mean_time:.2f}s\n"

            if 'total_gflops' in profiling:
                gflops = profiling['total_gflops']
                time_text += f"GFLOPs: {gflops:.0f}\n"

            if 'num_sam_prompts' in profiling:
                prompts = profiling['num_sam_prompts']
                time_text += f"SAM Prompts: {prompts}\n"

            ax.text(0.1, 0.5, time_text, fontsize=10,
                   verticalalignment='center', family='monospace')
            ax.set_title('Profiling', fontsize=12, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Guardado en: {output_path}")
    else:
        plt.show()

    plt.close()


def create_summary_table(configs):
    """
    Crea tabla comparativa de métricas de todas las configuraciones.
    """
    print("\n" + "="*120)
    print(f"{'Configuración':<30} {'mIoU':>10} {'Pixel Acc':>12} {'F1':>10} {'GFLOPs':>12} {'Prompts':>10} {'Time/img':>12}")
    print("="*120)

    for config_name, config_path in configs.items():
        results, _ = load_results(config_path)

        miou = results.get('miou', 0) * 100
        pixel_acc = results.get('pixel_accuracy', 0) * 100
        f1 = results.get('f1', 0) * 100

        profiling = results.get('profiling', {})
        gflops = profiling.get('total_gflops', 0)
        prompts = profiling.get('num_sam_prompts', 0)

        summary = profiling.get('summary', {})
        mean_time = 0
        if 'total_inference' in summary:
            mean_time = summary['total_inference'].get('mean_time', 0)

        print(f"{config_name:<30} {miou:>9.2f}% {pixel_acc:>11.2f}% {f1:>9.2f}% {gflops:>11.0f} {prompts:>10} {mean_time:>11.2f}s")

    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comparar resultados de segmentación visualmente')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Pares nombre:path de configuraciones (ej: baseline:results/baseline hybrid:results/hybrid)')
    parser.add_argument('--image-idx', type=int, default=0,
                       help='Índice de imagen a visualizar (default: 0)')
    parser.add_argument('--dataset', type=str, default='coco-stuff',
                       help='Nombre del dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Path para guardar visualización')
    parser.add_argument('--summary', action='store_true',
                       help='Mostrar tabla resumen de métricas')

    args = parser.parse_args()

    # Parsear configuraciones (formato: nombre:path)
    configs = {}
    for config_str in args.configs:
        if ':' not in config_str:
            print(f"Error: Formato incorrecto para config '{config_str}'. Use nombre:path")
            continue
        name, path = config_str.split(':', 1)
        configs[name] = path

    if not configs:
        print("Error: No se especificaron configuraciones válidas")
        return

    # Mostrar tabla resumen
    if args.summary:
        create_summary_table(configs)

    # Crear visualización comparativa
    compare_segmentations(
        args.image_idx,
        configs,
        dataset_name=args.dataset,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
