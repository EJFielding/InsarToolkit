KMLtemplate='''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
	<name>%s.kml</name>
	<StyleMap id="msn_ylw-pushpin">
		<Pair>
			<key>normal</key>
			<styleUrl>#sn_ylw-pushpin</styleUrl>
		</Pair>
		<Pair>
			<key>highlight</key>
			<styleUrl>#sh_ylw-pushpin</styleUrl>
		</Pair>
	</StyleMap>
	<Style id="sn_ylw-pushpin">
		<IconStyle>
			<scale>1.1</scale>
			<Icon>
				<href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
			</Icon>
			<hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>
		</IconStyle>
		<BalloonStyle>
		</BalloonStyle>
		<LineStyle>
			<color>ff272726</color>
		</LineStyle>
		<PolyStyle>
			<color>80555554</color>
			<outline>1</outline>
		</PolyStyle>
	</Style>
	<Style id="sh_ylw-pushpin">
		<IconStyle>
			<scale>1.3</scale>
			<Icon>
				<href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
			</Icon>
			<hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>
		</IconStyle>
		<BalloonStyle>
		</BalloonStyle>
		<LineStyle>
			<color>ff272726</color>
		</LineStyle>
		<PolyStyle>
			<color>80555554</color>
			<outline>0</outline>
		</PolyStyle>
	</Style>
	<Placemark>
		<name>SimpleRectangleGrey</name>
		<open>1</open>
		<styleUrl>#msn_ylw-pushpin</styleUrl>
		<Polygon>
			<tessellate>1</tessellate>
			<outerBoundaryIs>
				<LinearRing>
					<coordinates>
						%s
					</coordinates>
				</LinearRing>
			</outerBoundaryIs>
		</Polygon>
	</Placemark>
</Document>
</kml>
'''

class Polygon2KML:
	# Establish coordinates
	def __init__(self,name,polygon):
		self.name=name
		from shapely.geometry import Polygon
		# Polygon parameters
		self.polygon=polygon
		self.xCoords,self.yCoords=polygon.exterior.xy
		self.nPtsFull=len(self.xCoords) # n+1 vertices
		# Format coordinate string
		self.coords='' # empty coordinate string
		for i in range(self.nPtsFull):
			self.coords+='%f,%f,0 ' % \
			(self.xCoords[i],self.yCoords[i])
	# Write to text file
	def write(self,outFilename):
		outData=KMLtemplate % (self.name,self.coords)
		Fout=open(outFilename,'w')
		Fout.write(outData)
		Fout.close()
		print('Saved to: %s' % (outFilename))