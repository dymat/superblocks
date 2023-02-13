import React, {useState, useRef} from 'react';
import Dashboard from './dashboard/Dashboard'
import {MapContainer, TileLayer, FeatureGroup, GeoJSON} from "react-leaflet";
import {EditControl} from "react-leaflet-draw"

function App() {
    const [roi, setRoi] = useState([])
    const fgRef = useRef(null)
    const [currentLayer, setCurrentLayer] = useState(null)
    const [currentResponse, setCurrentResponse] = useState({})
    const [showResults, setShowResults] = useState({
        blocks: true,
        blocks_no_street: false,
        streets: false,
        buildings: false
    })
    const [highlightedBlockId, setHighlightedBlockId] = useState(-1)
    const [selectedSuperblock, setSelectedSuperblock] = useState(null)
    const [appStatus, setAppStatus] = useState("ready")
    const [stepperState, setStepperState] = useState(0)

    const handleStopDraw = e => {
        /*
        * read out the vertex coordinates of the shape drawn on the map
        * */

        const layers = e.target?._targets;

        if (layers && Object.keys(layers).length > 1) {
            const [key, lastLayer] = Object.entries(layers).pop()
            if (lastLayer?._latlngs?.length > 0) {
                if (setRoi) {
                    const coords_list = [...lastLayer._latlngs].pop().map(latlng => ({
                        lat: latlng.lat,
                        lng: latlng.lng
                    }))
                    setRoi(coords_list)
                    setCurrentLayer(lastLayer)
                }
            }
        }
    }

    const colors = ["#D32F2F", "#7B1FA2", "#1976D2", "#0097A7", "#388E3C", "#AFB42B", "#FFA000", "#E64A19"]

    const handleStartAnalyze = () => {

        setAppStatus('loading')

        const data = {
            name: "Berlin",
            coords: roi
        }

        fetch("http://localhost:9123/region", {
            method: "POST",
            mode: "cors",
            headers: {
                'Content-Type': 'application/json'
            },
            redirect: 'follow',
            referrerPolicy: 'no-referrer',
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(json => setCurrentResponse(json))
            .then(() => setAppStatus('ready'))
            .then(() => setStepperState(prevState => prevState + 1))
            .catch(() => setAppStatus("error"))
    }


    if (currentLayer && Object.keys(currentResponse).length > 0) {
        currentLayer.bringToBack()
        currentLayer.setStyle({fillOpacity: 0.02, weight: 3})
    }

    const renderSuperblocks = (key) => {
        if (!showResults[key] || !Object.keys(currentResponse).includes(key)) {
            return null
        }

        const shapes = currentResponse[key].features.map(item => {
            const blockId = item.properties.inter_id
            const style = {
                color: colors[blockId % colors.length],
                weight: highlightedBlockId === blockId ? 5 : 2,
                fillOpacity: highlightedBlockId === blockId ? 0.8 : 0.3
            }

            return <GeoJSON data={item} key={`${key}-${blockId}`} style={style} />
        })

        return shapes
    }

    const handleNextStep = () => {

        if (stepperState === 0) {
            handleStartAnalyze()
        } else {
            setStepperState(prevState => prevState + 1)
        }
    }

    const handleBackStep = () => {
        setStepperState( prevState => prevState - 1)
    }

    const handleReset = () => {
        setStepperState(0)
        setCurrentResponse({})
        setRoi([])
        setCurrentLayer(null)
        setSelectedSuperblock(null)
        fgRef.current.clearLayers()
    }


    return <Dashboard
        analyze={handleStartAnalyze}
        data={currentResponse}
        onHoverOverListItem={setHighlightedBlockId}
        selectSuperblock={setSelectedSuperblock}
        appStatus={appStatus}
        stepperState={stepperState}
        handleNextStep={handleNextStep}
        handleBackStep={handleBackStep}
        handleReset={handleReset}
    >
        <MapContainer center={[52.509, 13.385]}
                      zoom={13}
                      scrollWheelZoom={true}
                      zoomControl={false}
                      style={{display: "flex", width: "100%", minHeight: 800, height: "100%"}}>
            <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <FeatureGroup
                ref={fgRef}
            >
                <EditControl
                    draw={{
                        polygon: true,
                        polyline: false,
                        circle: false,
                        marker: false,
                        circlemarker: false
                    }}
                    edit={{
                        edit: false,
                        remove: false
                    }}
                    position={'topright'}
                    onDrawStart={() => fgRef.current.clearLayers()}
                    onDrawStop={handleStopDraw}
                    onEditStop={handleStopDraw}
                />
            </FeatureGroup>

            <FeatureGroup>

                {

                    renderSuperblocks('blocks')

                }
            </FeatureGroup>

        </MapContainer>
    </Dashboard>
}

export default App;
