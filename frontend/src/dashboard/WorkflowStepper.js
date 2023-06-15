import * as React from 'react';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import StepContent from '@mui/material/StepContent';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';

const steps = [
    {
        label: 'In welchem Bereich auf der Karte soll nach potentiellen Superblock gesucht werden?',
        description: `Markieren Sie auf der Karte das Gebiet, in dem nach Superblocks gesucht werden soll.
        Verwenden Sie dazu eines der Zeichenwerkzeuge rechts oben.`,
    },
    {
        label: 'Möchten Sie noch ein eigene Blocks hinzufügen?',
        description:
            'Um die Liste der gefundenen Superblocks durch eigene Blocks zu ergänzen, nutzen Sie bitte das' +
            'Polygon-Zeichenwerkzeug rechts oben. #TODO',
    },
    {
        label: 'Welche Kriterien für die qualtitative Bewertung der potentiellen Superblocks sind Ihnen wichtig?',
        description: `Lorem ipsum dolor sit amet. #TODO`,
    },
];


export default function WorkflowStepper(props) {
    const [calcTime, setCalcTime] = React.useState(-1);
    return (
        <Box>
            <Stepper activeStep={props.stepperState} orientation="vertical">
                {steps.map((step, index) => (
                    <Step key={step.label}>
                        <StepLabel
                            sx={{userSelect: 'none'}}
                        >
                            {step.label}
                        </StepLabel>
                        <StepContent>
                            <Typography sx={{userSelect: 'none'}}>
                                {props.stepperState === 1 ? 
                                <span>
                                    Berechnungszeit {(performance.now() - calcTime)/1000} Sekunden
                                    <br></br><br></br>
                                </span> : null}
                                {step.description}
                                {props.stepperState === 0
                                    ? <span style={{ display: 'inline-flex'}}>
                                        <span style={{width: 5}}></span>
                                        <img src="polygon.png" style={{height: 15, width: 13}} />
                                        <span style={{width: 5}}></span>
                                        <img src="rectangle.png" style={{height: 15, width: 13}} />
                                </span> : null}
                            </Typography>
                            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'right' }}>
                                <div>
                                    <Button
                                        disabled={index === 0}
                                        onClick={props.handleBackStep}
                                        sx={{ mt: 1, mr: 1 }}
                                    >
                                        Zurück
                                    </Button>

                                    <LoadingButton
                                        loading={props.appStatus === "loading"}
                                        loadingPosition="end"
                                        variant="contained"
                                        onClick =  {() => {
                                            setCalcTime(performance.now());
                                            props.handleNextStep();
                                        }}
                                        sx={{ mt: 1, mr: 1 }}
                                    >
                                        {index === steps.length - 1 ? 'Fertig' : 
                                            props.appStatus === "loading" && props.stepperState === 0? "Berechne Superblocks..." : 'Weiter'}
                                    </LoadingButton>
                                </div>
                            </Box>
                        </StepContent>
                    </Step>
                ))}
            </Stepper>
            {props.stepperState === steps.length && (
                <Paper square elevation={0} sx={{ p: 3, display: "flex", justifyContent: "left" }}>
                    <Button
                        variant="outlined" color="error"
                        onClick={props.handleReset}
                        sx={{ mt: 1, mr: 1 }}
                    >
                        Alles auf Anfang
                    </Button>
                </Paper>
            )}
        </Box>
    );
}