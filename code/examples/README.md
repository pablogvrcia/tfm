# Examples

This directory contains test images and videos for demonstrating CLIP-guided segmentation capabilities.

## Directory Structure

- `examples/` - Source images and videos
- `examples_results/` - Output segmentation results

## Test Cases

### MotoGP Racing

Segment motorcycle racing scenes with riders and environment.

**Single Frame:**
```bash
python clip_guided_segmentation.py \
    --image examples/motogp_frame.png \
    --vocabulary "Valentino Rossi Yamaha" "Marc Marquez Honda" track background grass \
    --output examples_results/motogp_frame
```

<table>
<tr>
<td width="50%">

**Input**

![MotoGP Input](motogp_frame.png)

</td>
<td width="50%">

**Output**

![MotoGP Result](../examples_results/motogp_frame.png)

</td>
</tr>
</table>

**Video:**
```bash
python clip_guided_segmentation.py \
    --image examples/motogp_video.mp4 \
    --vocabulary "Valentino Rossi Yamaha" "Marc Marquez Honda" track background grass \
    --output examples_results/motogp_video
```

- **Classes:** Riders (Rossi, Marquez), track, background, grass

---

### NBA Basketball

Segment basketball players and court elements.

**Single Frame:**
```bash
python clip_guided_segmentation.py \
    --image examples/nba_frame.png \
    --vocabulary "Stephen Curry" "LeBron James" floor crowd background \
    --output examples_results/nba_frame
```

<table>
<tr>
<td width="50%">

**Input**

![NBA Input](nba_frame.png)

</td>
<td width="50%">

**Output**

![NBA Result](../examples_results/nba_frame.png)

</td>
</tr>
</table>

**Video:**
```bash
python clip_guided_segmentation.py \
    --image examples/nba_video.mp4 \
    --vocabulary "Stephen Curry" "LeBron James" floor crowd background \
    --output examples_results/nba_video
```

- **Classes:** Players (Curry, LeBron), floor, crowd, background

---

### Football/Soccer

Segment football players from FC Barcelona.

```bash
python clip_guided_segmentation.py \
    --image examples/football_frame.png \
    --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
    --output examples_results/football_frame
```

<table>
<tr>
<td width="50%">

**Input**

![Football Input](football_frame.png)

</td>
<td width="50%">

**Output**

![Football Result](../examples_results/football_frame.png)

</td>
</tr>
</table>

- **Classes:** Players (Messi, Suarez, Neymar), grass, crowd, background

---

### Celebrity Recognition

Segment famous people in public events.

```bash
python clip_guided_segmentation.py \
    --image examples/obama_jordan.png \
    --vocabulary "Obama" "Michael Jordan" people background \
    --output examples_results/obama_jordan
```

<table>
<tr>
<td width="50%">

**Input**

![Obama Jordan Input](obama_jordan.png)

</td>
<td width="50%">

**Output**

![Obama Jordan Result](../examples_results/obama_jordan.png)

</td>
</tr>
</table>

- **Classes:** Obama, Michael Jordan, people, background

---

### Product/Brand Segmentation

Segment shoes from different brands.

```bash
python clip_guided_segmentation.py \
    --image examples/brands.png \
    --vocabulary "Nike Shoe" "Adidas Sneaker" background \
    --output examples_results/brands
```

<table>
<tr>
<td width="50%">

**Input**

![Brands Input](brands.png)

</td>
<td width="50%">

**Output**

![Brands Result](../examples_results/brands.png)

</td>
</tr>
</table>

- **Classes:** Nike shoes, Adidas sneakers, background

---

### F1 Podium Ceremony

Segment F1 drivers and podium elements.

```bash
python clip_guided_segmentation.py \
    --image examples/podium.png \
    --vocabulary champagne background hat "red bull driver" "Lewis Hamilton" podium hand \
    --output examples_results/podium
```

<table>
<tr>
<td width="50%">

**Input**

![Podium Input](podium.png)

</td>
<td width="50%">

**Output**

![Podium Result](../examples_results/podium.png)

</td>
</tr>
</table>

- **Classes:** Champagne, background, hat, Red Bull driver, Lewis Hamilton, podium, hand

---

## Features Demonstrated

### Image Segmentation
- Multi-class segmentation with distinct colors
- CLIP-guided prompt placement (intelligent vs blind grid)
- Class-specific filtering
- High-quality SAM masks

### Video Segmentation
- CLIP analysis on first frame only
- SAM2 temporal tracking across all frames
- Consistent object tracking throughout video
- H.264 encoding with faststart for compatibility

### Visualization
- Distinct color palette for each class
- Color legend showing class-to-color mapping
- Individual labels on larger objects (>1000 pixels)
- Adjustable mask opacity (70% by default)

## Output Formats

- **Images:** PNG format with high DPI (150)
- **Videos:** MP4 with H.264 codec, optimized for streaming
- All outputs include segmentation masks with colored overlays

## Notes

- The vocabulary can include specific person names, objects, or general categories
- CLIP intelligently identifies semantic concepts without training
- SAM2 provides high-quality mask boundaries
- Video processing uses CPU offloading to handle GPU memory constraints
