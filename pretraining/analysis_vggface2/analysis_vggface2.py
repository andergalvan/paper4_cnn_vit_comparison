import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict
import warnings
import time
from tqdm import tqdm
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['figure.max_open_warning'] = 0

SPLIT_DISPLAY_NAMES = {'train': 'Train', 'test': 'Test'}

PLOT_STYLE = {
    'figsize_class_dist': (18, 6),
    'figsize_image_props': (15, 10),
    'fontsize_axes': 16,
    'fontsize_title': 18,
    'fontsize_legend': 16,
    'dpi': 600,
    'colors': ['skyblue', 'lightcoral'],
    'edgecolor': 'black'
}

def format_number(x, pos=None):
    """Formato: miles con coma y decimales con punto"""
    if x >= 1000:
        return f"{x:,.0f}".replace(",", ",")
    else:
        return f"{x:,.1f}".replace(",", ",")

def analyze_dataset_structure(base_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    datasets = {'train': os.path.join(base_path, 'train'),
                'test': os.path.join(base_path, 'test')}
    for split_name, path in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️  WARNING: The {split_name} directory does not exist: {path}")
            return None
    return datasets

def count_images_per_class(dataset_path, split_name):
    class_counts = defaultdict(int)
    total_images = 0
    corrupted_images = []
    if not os.path.exists(dataset_path):
        return class_counts, total_images, corrupted_images

    all_image_paths = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    all_image_paths.append((class_name, os.path.join(class_path, f)))

    pbar = tqdm(all_image_paths, desc=f"Analysing {split_name}", unit="img")
    for class_name, image_path in pbar:
        try:
            with Image.open(image_path) as img:
                img.verify()
            class_counts[class_name] += 1
            total_images += 1
        except Exception as e:
            corrupted_images.append({'split': split_name, 'class': class_name, 'file': os.path.basename(image_path), 'path': image_path, 'error': str(e)})
    pbar.close()
    return dict(class_counts), total_images, corrupted_images

def analyze_image_properties(dataset_path, split_name, sample_size=200):
    image_info = []
    all_paths = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    all_paths.append((class_name, os.path.join(class_path, f)))
    if len(all_paths) > sample_size:
        sampled_paths = [all_paths[i] for i in np.random.choice(len(all_paths), sample_size, replace=False)]
    else:
        sampled_paths = all_paths

    for class_name, image_path in tqdm(sampled_paths, desc=f"Sampling {split_name}", unit="img"):
        try:
            with Image.open(image_path) as img:
                image_info.append({'split': split_name, 'class': class_name, 'filename': os.path.basename(image_path),
                                   'width': img.width, 'height': img.height, 'mode': img.mode, 'format': img.format,
                                   'size_bytes': os.path.getsize(image_path)})
        except:
            continue
    return pd.DataFrame(image_info)

def plot_class_distribution(class_counts_dict, datasets_info, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=PLOT_STYLE['figsize_class_dist'])

    splits = ['train', 'test']
    colors = PLOT_STYLE['colors']
    for i, split_name in enumerate(splits):
        counts = list(class_counts_dict.get(split_name, {}).values())
        ax = axes[i]
        if counts:
            ax.hist(counts, bins=20, alpha=0.7, edgecolor=PLOT_STYLE['edgecolor'], color=colors[i])
            ax.set_xlabel('Number of Images per Class', fontsize=PLOT_STYLE['fontsize_axes'])
            ax.set_ylabel('Frequency', fontsize=PLOT_STYLE['fontsize_axes'])
            ax.set_title(f'{SPLIT_DISPLAY_NAMES[split_name]} Class Distribution', 
                         fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')  # <- Título en negrita
            ax.tick_params(axis='x', labelsize=PLOT_STYLE['fontsize_axes'])
            ax.tick_params(axis='y', labelsize=PLOT_STYLE['fontsize_axes'])
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_number))
            mean_count = np.mean(counts)
            median_count = np.median(counts)
            ax.axvline(mean_count, color='red', linestyle='--', label=f"Mean: {format_number(mean_count)}")
            ax.axvline(median_count, color='green', linestyle='--', label=f"Median: {format_number(median_count)}")
            ax.legend(fontsize=PLOT_STYLE['fontsize_legend'])
        else:
            ax.axis('off')

    totals = [datasets_info[s]['total_images'] for s in splits]
    ax_totals = axes[2]
    ax_totals.bar([SPLIT_DISPLAY_NAMES[s] for s in splits], totals, color=colors, edgecolor=PLOT_STYLE['edgecolor'])
    ax_totals.set_ylabel('Number of Images', fontsize=PLOT_STYLE['fontsize_axes'])
    ax_totals.set_title('Total Images per Split', fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')  # <- Título en negrita
    ax_totals.tick_params(axis='x', labelsize=PLOT_STYLE['fontsize_axes'])
    ax_totals.tick_params(axis='y', labelsize=PLOT_STYLE['fontsize_axes'])
    ax_totals.yaxis.set_major_formatter(mtick.FuncFormatter(format_number))
    for i, v in enumerate(totals):
        ax_totals.text(i, v + max(totals)*0.01, format_number(v), ha='center', fontsize=PLOT_STYLE['fontsize_axes'], fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close()
    return [plot_path]

def plot_image_properties(image_properties, output_dir):
    if image_properties.empty:
        return []

    fig, axes = plt.subplots(2, 2, figsize=PLOT_STYLE['figsize_image_props'])

    resolutions = image_properties['width'].astype(str) + 'x' + image_properties['height'].astype(str)
    top_resolutions = resolutions.value_counts().head(10)
    axes[0, 0].bar(range(len(top_resolutions)), top_resolutions.values, color='skyblue', edgecolor=PLOT_STYLE['edgecolor'])

    axes[0, 0].set_xticks(range(len(top_resolutions)))
    axes[0, 0].set_xticklabels(top_resolutions.index, rotation=45, ha='right')

    axes[0, 0].set_xlabel('Resolution', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[0, 0].set_ylabel('Frequency', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[0, 0].set_title('Top 10 Image Resolutions', fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')
    axes[0, 0].tick_params(axis='y', labelsize=PLOT_STYLE['fontsize_axes'])
    axes[0, 0].yaxis.set_major_formatter(mtick.FuncFormatter(format_number))


    fmt_counts = image_properties['format'].value_counts()
    axes[0, 1].pie(fmt_counts.values, labels=fmt_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    axes[0, 1].set_title('Image Formats Distribution', fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')

    sizes_mb = image_properties['size_bytes'] / (1024*1024)
    axes[1, 0].hist(sizes_mb, bins=30, alpha=0.7, edgecolor=PLOT_STYLE['edgecolor'], color='lightgreen')
    axes[1, 0].set_xlabel('Size (MB)', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 0].set_ylabel('Frequency', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 0].set_title('Image Size Distribution', fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')
    axes[1, 0].tick_params(axis='x', labelsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 0].tick_params(axis='y', labelsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 0].yaxis.set_major_formatter(mtick.FuncFormatter(format_number))

    axes[1, 1].scatter(image_properties['width'], image_properties['height'], alpha=0.6, color='coral')
    axes[1, 1].set_xlabel('Width (pixels)', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 1].set_ylabel('Height (pixels)', fontsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 1].set_title('Width vs Height Scatter Plot', fontsize=PLOT_STYLE['fontsize_title'], fontweight='bold')
    axes[1, 1].tick_params(axis='x', labelsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 1].tick_params(axis='y', labelsize=PLOT_STYLE['fontsize_axes'])
    axes[1, 1].yaxis.set_major_formatter(mtick.FuncFormatter(format_number))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'image_properties.png')
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close()
    return [plot_path]

def analyze_class_overlap(class_counts_dict):
    train_classes = set(class_counts_dict.get('train', {}).keys())
    test_classes = set(class_counts_dict.get('test', {}).keys())
    train_test_overlap = train_classes.intersection(test_classes)
    return {'train_classes': train_classes, 'test_classes': test_classes, 'overlaps': {'train_test': train_test_overlap}}

def generate_summary_report(datasets_info, class_counts_dict, corrupted_images, image_properties):
    total_images = sum(info['total_images'] for info in datasets_info.values())
    total_classes = sum(len(counts) for counts in class_counts_dict.values())
    total_corrupted = len(corrupted_images)
    return {'total_images': total_images, 'total_classes': total_classes, 'corrupted_count': total_corrupted}

def main_analysis(base_path, output_dir):
    start_time = time.time()
    datasets = analyze_dataset_structure(base_path, output_dir)
    if datasets is None:
        return

    class_counts_dict = {}
    datasets_info = {}
    all_corrupted_images = []
    for split_name, dataset_path in datasets.items():
        class_counts, total_images, corrupted = count_images_per_class(dataset_path, split_name)
        class_counts_dict[split_name] = class_counts
        datasets_info[split_name] = {'path': dataset_path, 'total_images': total_images, 'total_classes': len(class_counts)}
        all_corrupted_images.extend(corrupted)

    all_image_properties = []
    for split_name, dataset_path in datasets.items():
        props = analyze_image_properties(dataset_path, split_name, sample_size=200)
        all_image_properties.append(props)
    image_properties = pd.concat(all_image_properties, ignore_index=True)

    overlap_analysis = analyze_class_overlap(class_counts_dict)
    plot_paths = plot_class_distribution(class_counts_dict, datasets_info, output_dir)
    if not image_properties.empty:
        plot_paths.extend(plot_image_properties(image_properties, output_dir))

    summary = generate_summary_report(datasets_info, class_counts_dict, all_corrupted_images, image_properties)
    end_time = time.time()
    print(f"\n✅ Analysis completed in {end_time - start_time:.1f}s. Results in: {output_dir}")
    return {'datasets_info': datasets_info, 'class_counts': class_counts_dict, 'corrupted_images': all_corrupted_images,
            'image_properties': image_properties, 'overlap_analysis': overlap_analysis, 'summary': summary,
            'saved_files': {'plots': plot_paths, 'output_directory': output_dir}}

if __name__ == "__main__":
    BASE_PATH = "/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/vggface2"
    OUTPUT_DIR = "/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/analysis_vggface2"
    results = main_analysis(BASE_PATH, OUTPUT_DIR)