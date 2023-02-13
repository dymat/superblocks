import * as React from 'react';
import {styled, createTheme, ThemeProvider} from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import MuiDrawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import MuiAppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ModeStandbyRoundedIcon from '@mui/icons-material/ModeStandbyRounded';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ListItemIcon from "@mui/material/ListItemIcon";
import DashboardIcon from "@mui/icons-material/Dashboard";
import ListItemText from "@mui/material/ListItemText";
import ListItemButton from "@mui/material/ListItemButton";
import ListSubheader from "@mui/material/ListSubheader";
import Popper from "@mui/material/Popper";

import CenteredTabs from './AppBarTabs'
import ControlledAccordions from './Accordion'

const drawerWidth = 240;

const AppBar = styled(MuiAppBar, {
    shouldForwardProp: (prop) => prop !== 'open',
})(({theme, open}) => ({
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    }),
}));

const Drawer = styled(MuiDrawer, {shouldForwardProp: (prop) => prop !== 'open'})(
    ({theme, open}) => ({
        '& .MuiDrawer-paper': {
            position: 'relative',
            whiteSpace: 'nowrap',
            width: drawerWidth,
            transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
            }),
            boxSizing: 'border-box',
            ...(!open && {
                overflowX: 'hidden',
                transition: theme.transitions.create('width', {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.leavingScreen,
                }),
                width: theme.spacing(7),
                [theme.breakpoints.up('sm')]: {
                    width: theme.spacing(9),
                },
            }),
        },
    }),
);

const AccordionContainer = styled(Container, {shouldForwardProp: (prop) => prop !== 'open'})(
    ({theme, open}) => ({
        position: 'absolute',
        zIndex: theme.zIndex.appBar - 10,
        padding: 0,
        width: 600,
        top: 64 + 24,
        left: open ? drawerWidth : 72,
        transition: theme.transitions.create('left', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    }),
);

const mdTheme = createTheme();

function Dashboard(props) {
    const [open, setOpen] = React.useState(false);
    const toggleDrawer = () => {
        setOpen(!open);
    };


    const listItems = () => {
        if (!props.data || !props.data.blocks) {
            return null
        }

        const miniblocks = props.data.blocks.features.filter(b => b.properties.b_type.toLowerCase() === "miniblock")
        const superblocks = props.data.blocks.features.filter(b => b.properties.b_type.toLowerCase() === "superblock")

        const listMiniblocks = miniblocks.map(block => {

            const blockId = block.properties.inter_id

            return (
                <ListItemButton
                    key={`miniblock-list-item-${blockId}`}
                    onClick={() => props.selectSuperblock(blockId)}
                    onMouseOver={() => props.onHoverOverListItem(blockId)}
                    onMouseOut={() => props.onHoverOverListItem(-1)}
                >
                    <ListItemIcon>
                        <DashboardIcon/>
                    </ListItemIcon>
                    <ListItemText primary={`Miniblock #${blockId}`}/>
                </ListItemButton>)
        })

        const listSuperblocks = superblocks.map(block => {

            const blockId = block.properties.inter_id

            return (
                <ListItemButton
                    key={`superblock-list-item-${blockId}`}
                    onClick={() => props.selectSuperblock(blockId)}
                    onMouseOver={() => props.onHoverOverListItem(blockId)}
                    onMouseOut={() => props.onHoverOverListItem(-1)}
                >
                    <ListItemIcon>
                        <DashboardIcon/>
                    </ListItemIcon>
                    <ListItemText primary={`Superblock #${blockId}`}/>
                </ListItemButton>)
        })

        return <>

            {
                listSuperblocks.length > 0 ?
                    <ListSubheader component="div" inset>
                        Superblocks
                    </ListSubheader>
                    : null
            }
            {listSuperblocks}

            {
                listSuperblocks.length > 0 && listMiniblocks.length > 0 ?
                    <Divider/>
                    : null
            }


            {
                listMiniblocks.length > 0 ?
                    <ListSubheader component="div" inset>
                        Miniblocks
                    </ListSubheader>
                    : null
            }
            {listMiniblocks}
        </>
    }


    return (
        <ThemeProvider theme={mdTheme}>
            <Box sx={{display: 'flex'}}>
                <CssBaseline/>
                <AppBar position="absolute" open={open}>
                    <Toolbar
                        sx={{
                            pr: '24px', // keep right padding when drawer closed
                        }}
                    >
                        <IconButton
                            edge="start"
                            color="inherit"
                            aria-label="open drawer"
                            onClick={toggleDrawer}
                            sx={{
                                marginRight: '36px',
                                ...(open && {display: 'none'}),
                            }}
                        >
                            <MenuIcon/>
                        </IconButton>
                        <Typography
                            component="h1"
                            variant="h6"
                            color="inherit"
                            noWrap
                            sx={{flexGrow: 1}}
                        >
                            <CenteredTabs/>
                        </Typography>

                    </Toolbar>
                </AppBar>
                <Drawer variant="permanent" open={open}>
                    <Toolbar
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'flex-end',
                            px: [1],
                        }}
                    >
                        <IconButton onClick={toggleDrawer}>
                            <ChevronLeftIcon/>
                        </IconButton>
                    </Toolbar>
                    <Divider/>
                    <List component="nav">
                        {

                            listItems()

                        }
                    </List>
                </Drawer>
                <Box
                    component="main"
                    sx={{
                        backgroundColor: (theme) =>
                            theme.palette.mode === 'light'
                                ? theme.palette.grey[100]
                                : theme.palette.grey[900],
                        flexGrow: 1,
                        height: '100vh',
                        overflow: 'auto',
                    }}
                >
                    <Toolbar/>

                    { /* Map */}
                    <Box sx={{height: "calc(100% - 64px)"}}>
                        {props.children}
                    </Box>

                    { /* Accordions */}
                    <AccordionContainer open={open}>
                        <ControlledAccordions {...props} />
                    </AccordionContainer>


                </Box>
            </Box>
        </ThemeProvider>
    );
}

export default Dashboard;
