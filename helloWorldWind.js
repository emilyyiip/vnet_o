var wwd = new WorldWind.WorldWindow("canvasOne");
wwd.addLayer(new WorldWind.BMNGOneImageLayer());
wwd.addLayer(new WorldWind.BMNGLandsatLayer());
wwd.addLayer(new WorldWind.CompassLayer());
wwd.addLayer(new WorldWind.CoordinatesDisplayLayer(wwd));
wwd.addLayer(new WorldWind.ViewControlsLayer(wwd));

var placemarkLayer = new WorldWind.RenderableLayer("Placemark");
wwd.addLayer(placemarkLayer);
var placemarkAttributes = new WorldWind.PlacemarkAttributes(null);

placemarkAttributes.imageOffset = new WorldWind.Offset(
    WorldWind.OFFSET_FRACTION, 0.3,
    WorldWind.OFFSET_FRACTION, 0.0);

placemarkAttributes.labelAttributes.color = WorldWind.Color.YELLOW;
placemarkAttributes.labelAttributes.offset = new WorldWind.Offset(
            WorldWind.OFFSET_FRACTION, 0.5,
            WorldWind.OFFSET_FRACTION, 1.0);
placemarkAttributes.imageSource = WorldWind.configuration.baseUrl + "images/pushpins/plain-red.png";
var position = new WorldWind.Position(55.0, -106.0, 100.0);
var placemark = new WorldWind.Placemark(position, false, placemarkAttributes);
placemark.label = "Placemark\n" +
    "Lat " + placemark.position.latitude.toPrecision(4).toString() + "\n" +
    "Lon " + placemark.position.longitude.toPrecision(5).toString();
placemark.alwaysOnTop = true;
placemarkLayer.addRenderable(placemark);

var renderableLayer = new WorldWind.RenderableLayer("Path Layer");
wwd.addLayer(renderableLayer);

// read csv
async function parseCSV(url){
  let response = await fetch(url);
  let data = await response.text();
  return Papa.parse(data, {
    header: true, 
    dynamicTyping: true, 
    skipEmptyLines:true
  }).data;
  }

// plot a line
function plotLine(lat1, lon1, lat2, lon2){
  var pathPositions = [
    new WorldWind.Position(lat1, lon1, 0),
    new WorldWind.Position(lat2, lon2, 0) 
 ];
  
  var path = new WorldWind.Path(pathPositions, null);
  path.altitudeMode = WorldWind.CLAMP_TO_GROUND; // clamps to gound and not altitude
  path.followTerrain = true; 
  path.extrude = true;
  path.useSurfaceShapefor2D = true;
  path.attributes = new WorldWind.ShapeAttributes(null);
  path.attributes.color = WorldWind.Color.RED;
  path.attributes.interiorColor = new WorldWind.Color(1, 0.0, 0.0, 0.5);

  renderableLayer.addRenderable(path);
  wwd.addLayer(renderableLayer);
  wwd.redraw();
    }

async function procCSV(one_data, two_data){
  let one = await parseCSV(one_data);
  let two = await parseCSV(two_data);
  one.forEach(item1 => {
    let item2 = two.find(item => item.index === item1.index);
    if(item2){
      plotLine(item1.lat, item1.long, item2.lat, item2.long)
    }
  });
  wwd.redraw();
}
                          
const csv1 = '50000_dept.csv';
const csv2 = '50000_arr.csv';
procCSV(csv1, csv2);





