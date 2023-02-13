import * as React from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import WorkflowStepper from './WorkflowStepper'

export default function ControlledAccordions(props) {
    const [expanded, setExpanded] = React.useState('panel1');

    const handleChange = (panel) => (event, isExpanded) => {
        setExpanded(isExpanded ? panel : false);
    };

    return (
        <div>
            <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1bh-content"
                    id="panel1bh-header"
                >
                    <Typography sx={{ width: '33%', flexShrink: 0 }}>
                        Superblocks generieren
                    </Typography>
                    <Typography sx={{ color: 'text.secondary' }}>
                        Schritt-für-Schritt zum Superblock
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <WorkflowStepper {...props} />
                </AccordionDetails>
            </Accordion>
        </div>
    );
}
