python clip_guided_segmentation.py --image examples/motogp_frame.png --vocabulary "Valentino Rossi Yamaha" "Marc Marquez Honda" track background grass --output examples_results/motogp_frame

python clip_guided_segmentation.py --image examples/motogp_video.mp4 --vocabulary "Valentino Rossi Yamaha" "Marc Marquez Honda" track background grass --output examples_results/motogp_video

python clip_guided_segmentation.py --image examples/nba_frame.png --vocabulary "Stephen Curry" "LeBron James" floor crowd background --output examples_results/nba_frame

python clip_guided_segmentation.py --image examples/nba_video.mp4 --vocabulary "Stephen Curry" "LeBron James" floor crowd background --output examples_results/nba_video

python clip_guided_segmentation.py --image examples/football_frame.png --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background  --output examples_results/football_frame

python clip_guided_segmentation.py --image examples/congress.png --vocabulary "Obama" "Michael Jordan" people background --output examples_results/congress

python clip_guided_segmentation.py --image examples/brands.png --vocabulary "Nike Shoe" "Adidas Sneaker" background --output examples_results/brands