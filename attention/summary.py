import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os

def generate_attention_summary(spatial_maps, channel_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('CNN Attention Heatmap Summary', fontsize=22, fontweight='bold', color='#2a2a2a', y=0.98)
    avg_spatial = np.mean(spatial_maps, axis=0)
    im1 = axes[0,0].imshow(avg_spatial, cmap='jet')
    axes[0,0].set_title('High Activity Zones', fontsize=16, fontweight='bold', pad=10)
    axes[0,0].axis('off')
    max_idx = np.unravel_index(np.argmax(avg_spatial), avg_spatial.shape)
    car_icon_path = os.path.join(os.path.dirname(__file__), '../../static', 'car_icon.png')
    if os.path.exists(car_icon_path):
        car_img = mpimg.imread(car_icon_path)
        h, w = avg_spatial.shape
        icon_h, icon_w = car_img.shape[:2]
        extent = [max_idx[1]-icon_w//2, max_idx[1]+icon_w//2, max_idx[0]+icon_h//2, max_idx[0]-icon_h//2]
        axes[0,0].imshow(car_img, extent=extent, alpha=0.7, zorder=10)
    axes[0,0].text(10, 30, 'RISK ZONE DETECTED!', fontsize=18, color='white', fontweight='bold', bbox=dict(facecolor='red', edgecolor='white', boxstyle='round,pad=0.3'), zorder=11)
    axes[0,0].add_patch(mpatches.Rectangle((0.02, 0.88), 0.22, 0.09, transform=axes[0,0].transAxes, facecolor='white', edgecolor='black', zorder=12))
    axes[0,0].text(0.03, 0.92, 'Red = High Risk\nBlue = Safe', color='black', fontsize=11, transform=axes[0,0].transAxes, zorder=13)
    axes[0,0].text(0.5, -0.13, 'Bright spots show where accidents are likely ', fontsize=12, color='#444', ha='center', va='center', transform=axes[0,0].transAxes)
    var_spatial = np.var(spatial_maps, axis=0)
    cmap_conf = mcolors.LinearSegmentedColormap.from_list('custom_conf', ['#ffe600', '#001f4d'][::-1])
    im2 = axes[0,1].imshow(var_spatial, cmap=cmap_conf)
    axes[0,1].set_title('Confidence Hotspots', fontsize=16, fontweight='bold', pad=10)
    axes[0,1].axis('off')
    max_var_idx = np.unravel_index(np.argmax(var_spatial), var_spatial.shape)
    circ = mpatches.Circle((max_var_idx[1], max_var_idx[0]), radius=15, fill=False, edgecolor='red', linestyle='dashed', linewidth=2.5, zorder=10)
    axes[0,1].add_patch(circ)
    axes[0,1].text(max_var_idx[1]+18, max_var_idx[0], 'Uncertain Zone', color='red', fontsize=13, fontweight='bold', zorder=11)
    axes[0,1].add_patch(mpatches.Rectangle((0.02, 0.88), 0.32, 0.09, transform=axes[0,1].transAxes, facecolor='white', edgecolor='black', zorder=12))
    axes[0,1].text(0.03, 0.92, 'Yellow = Uncertain\nBlue = Confident', color='black', fontsize=11, transform=axes[0,1].transAxes, zorder=13)
    axes[0,1].text(0.5, -0.13, 'Yellow areas need review - Model less sure here.', fontsize=12, color='#444', ha='center', va='center', transform=axes[0,1].transAxes)
    channel_matrix = np.array(channel_data)
    im3 = axes[1,0].imshow(channel_matrix.T, aspect='auto', cmap='summer')
    axes[1,0].set_title('Feature Importance Timeline', fontsize=16, fontweight='bold', pad=10)
    axes[1,0].set_xlabel('Frame Number', fontsize=12)
    axes[1,0].set_ylabel('Feature', fontsize=12)
    axes[1,0].grid(visible=True, color='#e0e0e0', linestyle='--', linewidth=0.5, alpha=0.7)
    key_frame = int(np.argmax(np.sum(channel_matrix, axis=1)))
    axes[1,0].axvline(key_frame, color='yellow', linestyle='-', linewidth=2.5, zorder=10)
    axes[1,0].text(key_frame+1, 2, 'Key Moment', color='yellow', fontsize=13, fontweight='bold', zorder=11)
    axes[1,0].text(0.5, -0.18, 'Green peaks show critical features over time ', fontsize=12, color='#444', ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].annotate('Bright = More Important', xy=(1.01, 0.5), xycoords='axes fraction', fontsize=11, color='#444', rotation=90, va='center')
    avg_channel = np.mean(channel_matrix, axis=0)
    top_channels = np.argsort(avg_channel)[-10:]
    bars = axes[1,1].bar(range(1, 11), avg_channel[top_channels], color='#ff9800', edgecolor='#b8860b', linewidth=1.2)
    axes[1,1].set_title('Top Decision Drivers', fontsize=16, fontweight='bold', pad=10)
    axes[1,1].set_xlabel('Top Feature Rank', fontsize=12)
    axes[1,1].set_ylabel('Importance Score', fontsize=12)
    axes[1,1].set_xticks(range(1, 11))
    axes[1,1].set_xticklabels([str(i) for i in range(1, 11)], fontsize=10)
    axes[1,1].tick_params(axis='y', labelsize=10)
    axes[1,1].spines['top'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].spines['left'].set_color('#cccccc')
    axes[1,1].spines['bottom'].set_color('#cccccc')
    axes[1,1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1,1].grid(visible=True, axis='y', color='#e0e0e0', linestyle='--', linewidth=0.7, alpha=0.7)
    axes[1,1].axhline(0.6, color='red', linestyle='--', linewidth=2, zorder=10)
    axes[1,1].text(10.2, 0.6, 'High Risk Line', color='red', fontsize=11, va='center', fontweight='bold')
    max_idx = np.argmax(avg_channel[top_channels])
    axes[1,1].annotate('Most Attended ★', xy=(max_idx+1, avg_channel[top_channels][max_idx]), xytext=(max_idx+1, avg_channel[top_channels][max_idx]+0.05), ha='center', color='#b8860b', fontsize=13, fontweight='bold', arrowprops=dict(facecolor='#b8860b', shrink=0.05, width=1, headwidth=7))
    legend_patches = [mpatches.Patch(color='#ff9800', label='Key Drivers'), mpatches.Patch(color='red', label='High Risk Line')]
    axes[1,1].legend(handles=legend_patches, loc='upper right', fontsize=11, frameon=False)
    axes[1,1].text(0.5, -0.18, 'These features drive our accident predictions ', fontsize=12, color='#444', ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].annotate('Rank 1 = Most Influential', xy=(0.5, -0.13), xycoords='axes fraction', fontsize=11, color='#444', ha='center')
    axes[1,1].set_ylim(0, 1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=340, bbox_inches='tight')
    plt.close() 