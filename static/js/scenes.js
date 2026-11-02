// Captions: https://huggingface.co/datasets/AIGeeksGroup/PanDWM
// Source: 2_长视频_149个_v2.zip / 索引.csv, clip indices 75, 101 and 131.
// Use the newer local videos; only scene descriptions come from the archive.
window.PANDWM_SCENES = [
  {
    scene: 75,
    title: 'Industrial parking lot',
    caption: 'Rain, industrial, parked trucks at parking lot',
    nuscenesScene: 'scene-0633',
    duration: 19.2,
    fps: 10,
    directory: 'static/videos/scene75',
    posterPrefix: 'static/images/scene75'
  },
  {
    scene: 101,
    title: 'Rainy intersection',
    caption: 'Rain, parked cars, parked bicycle, peds, cones, cross intersection, parking lot, turning right',
    nuscenesScene: 'scene-0906',
    duration: 19.2,
    fps: 10,
    directory: 'static/videos/scene101',
    posterPrefix: 'static/images/scene101'
  },
  {
    scene: 131,
    title: 'Pedestrians and traffic',
    caption: 'Cones, cars, many peds, changing lane, parking lot, parked cars, parked motorcycles',
    nuscenesScene: 'scene-0969',
    duration: 19.2,
    fps: 10,
    directory: 'static/videos/scene131',
    posterPrefix: 'static/images/scene131'
  }
];
