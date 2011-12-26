<?php

add_action( 'admin_init', 'color_scheme_init' );
add_action( 'admin_menu', 'color_scheme_add_page' );

/**
 * Init plugin options to white list our options
 */
function color_scheme_init(){
	register_setting( 'theme_options', 'theme_color_scheme', 'theme_color_scheme_validate' );
	register_setting( 'theme_options', 'theme_credits', 'credits_validate' );
	register_setting( 'theme_options', 'author_credits', 'credits_validate' );
}

/**
 * Load up the menu page
 */
function color_scheme_add_page() {
	add_theme_page( __( 'Color Scheme' ), __( 'Color Scheme' ), 'edit_theme_options', 'color_scheme', 'color_scheme_do_page' );
}

/**
 * Create array for the Color Schemes
 */

$color_schemes = array(
	'gray' => array(
		'value' => 'gray',
		'label' => __( 'Gray' )
	),
	'red' => array(
		'value' => 'red',
		'label' => __( 'Red' )
	),
	'blue' => array(
		'value' => 'blue',
		'label' => __( 'Blue' )
	),
	'green' => array(
		'value' => 'green',
		'label' => __( 'Green' )
	),
	'ocean' => array(
		'value' => 'ocean',
		'label' => __( 'Ocean' )
	),
	'olive' => array(
		'value' => 'olive',
		'label' => __( 'Olive' )
	),
	'purple' => array(
		'value' => 'purple',
		'label' => __( 'Purple' )
	),
	'orange' => array(
		'value' => 'orange',
		'label' => __( 'Orange' )
	),
	'pink' => array(
		'value' => 'pink',
		'label' => __( 'Pink' )
	),
	'brown' => array(
		'value' => 'brown',
		'label' => __( 'Brown' )
	)
);

/**
 * Create the options page
 */
function color_scheme_do_page() {
	global $color_schemes;

	if ( ! isset( $_REQUEST['updated'] ) )
		$_REQUEST['updated'] = false;

	?>
	<div class="wrap">
		<?php screen_icon(); echo "<h2>" . get_current_theme() . __( ' Color Scheme' ) . "</h2>"; ?>

		<?php if ( false !== $_REQUEST['updated'] ) : ?>
		<div class="updated fade"><p><strong><?php _e( 'Options saved' ); ?></strong></p></div>
		<?php endif; ?>

		<form method="post" action="options.php">
			<?php settings_fields( 'theme_options' ); ?>
            <?php $options = get_option( 'theme_color_scheme', 'gray' );
			$author_credits = get_option( 'author_credits', false );
			$theme_credits = get_option( 'theme_credits', true ); ?>

			<table class="form-table">

				<tr valign="top"><th scope="row"><?php _e( 'Use the following color scheme' ); ?></th>
					<td>
						<fieldset><legend class="screen-reader-text"><span><?php _e( 'Use the following color scheme' ); ?></span></legend>
						<?php foreach( $color_schemes as $scheme ) : ?>
                            <input type="radio" id="<?php echo $scheme['value']; ?>" name="theme_color_scheme" value="<?php esc_attr_e( $scheme['value'] ); ?>" <?php checked( $options, $scheme['value'] ); ?> />
                            <label for="<?php echo $scheme['value']; ?>"><?php echo $scheme['label']; ?></label><br />
                        <?php endforeach; ?>

						</fieldset>
					</td>
				</tr>
                
                <tr valign="top"><th scope="row">Theme Credits</th>
                	<td>
						<fieldset><legend class="screen-reader-text"><span>Credits</span></legend>
                        
                        <input type="checkbox" id="theme_credits" name="theme_credits" value="1" <?php checked( true, $theme_credits ); ?> />
                        <label for="theme_credits">Show Theme Credits</label><br />
                        
                        <input type="checkbox" id="author_credits" name="author_credits" value="1" <?php checked( true, $author_credits ); ?> />
                        <label for="author_credits">Show Author Credits</label>
                    </td>
                </tr>
                
			</table>

			<p class="submit">
				<input type="submit" class="button-primary" value="<?php _e( 'Save Options' ); ?>" />
			</p>
		</form>
	</div>
	<?php
}

/**
 * Sanitize and validate input.
 */
function theme_color_scheme_validate( $input ) {
	global $color_schemes;

	// Our color scheme option must actually be in our array of color schemes
	if ( ! isset( $input ) )
		$input = null;
	if ( ! array_key_exists( $input, $color_schemes ) )
		$input = null;

	return $input;
}

function credits_validate( $input ) {
	// If the checkbox has not been checked, we void it
	if ( ! isset( $input ) )
		$input = null;
	// We verify if the input is a boolean value
	$input = ( $input == 1 ? 1 : 0 );
	return $input;
}

// adapted from http://planetozh.com/blog/2009/05/handling-plugins-options-in-wordpress-28-with-register_setting/

function color_scheme() {
	$options = get_option('theme_color_scheme', 'gray');
	switch ($options) {
	  case 'red':
	    $bodybg = '220000';
		$containerbg = '2c0000';
		$containerbd = '350000';
		$elementbg = '340000';
		$elementbd = '450000';
		$elementhvbg = '3b0000';
		$elementhvbd = '510000';
	    break;
	  case 'blue':
	    $bodybg = '000022';
		$containerbg = '00002c';
		$containerbd = '000035';
		$elementbg = '000034';
		$elementbd = '000045';
		$elementhvbg = '00003b';
		$elementhvbd = '000051';
	    break;
	  case 'green':
	    $bodybg = '002200';
		$containerbg = '002c00';
		$containerbd = '003500';
		$elementbg = '003400';
		$elementbd = '004500';
		$elementhvbg = '003b00';
		$elementhvbd = '005100';
	    break;
	  case 'ocean':
	    $bodybg = '002222';
		$containerbg = '002c2c';
		$containerbd = '003535';
		$elementbg = '003434';
		$elementbd = '004545';
		$elementhvbg = '003b3b';
		$elementhvbd = '005151';
	    break;
	  case 'olive':
	    $bodybg = '222200';
		$containerbg = '2c2c00';
		$containerbd = '353500';
		$elementbg = '343400';
		$elementbd = '454500';
		$elementhvbg = '3b3b00';
		$elementhvbd = '515100';
	    break;
	  case 'purple':
	    $bodybg = '220022';
		$containerbg = '2c002c';
		$containerbd = '350035';
		$elementbg = '340034';
		$elementbd = '450045';
		$elementhvbg = '3b003b';
		$elementhvbd = '510051';
	    break;
	  case 'orange':
	    $bodybg = 'cc6600';
		$containerbg = 'cf6f00';
		$containerbd = 'df7f00';
		$elementbg = 'dd7d00';
		$elementbd = 'ff8f00';
		$elementhvbg = 'f48400';
		$elementhvbd = 'ff9900';
	    break;
	  case 'pink':
	    $bodybg = 'cc0066';
		$containerbg = 'cf006f';
		$containerbd = 'df007f';
		$elementbg = 'dd007d';
		$elementbd = 'ff008f';
		$elementhvbg = 'f40084';
		$elementhvbd = 'ff0099';
	    break;
	  case 'brown':
	    $bodybg = '884400';
		$containerbg = '8f4f00';
		$containerbd = '7f3f00';
		$elementbg = '9d5d00';
		$elementbd = 'af6f00';
		$elementhvbg = 'a56500';
		$elementhvbd = 'af7700';
	    break;
	}
	
	if ($options != 'gray') : ?>
<style type="text/css">
/* Custom Color Scheme */
body {
	background:#<?php echo $bodybg; ?>;
}
.container, #top-nav li.current_page_item a, #top-nav li.current-menu-item a, #searchsubmit, pre {
	background:#<?php echo $containerbg; ?>;
	border:1px #<?php echo $containerbd; ?> solid;
}
pre {
	border:none;
}
#top-nav li.current_page_item a, #top-nav li.current-menu-item a {
	border-bottom:none;
}
.element, #top-nav ul li, ul#top-nav ul li a, .widget li, #s {
	background:#<?php echo $elementbg; ?>;
	border-bottom:1px #<?php echo $elementbd; ?> solid;
}
#s {
	border:1px #<?php echo $elementbd; ?> solid;
}
.element:hover, #top-nav ul li:hover, .widget li:hover, input#s:focus {
	background:#<?php echo $elementhvbg; ?>;
	border-bottom-color:#<?php echo $elementhvbd; ?>;
}
.sticky {
	border:2px #<?php echo $elementbd; ?> solid;
	border-top-width:1px;
}
#accordion li.first a, .widget ul {
	border-top:1px #<?php echo $elementbd; ?> solid;
}
.navigation a {
	background:#<?php echo $elementbg; ?>;
	border:1px #<?php echo $elementbd; ?> solid;
}
.navigation a:hover {
	background:#<?php echo $elementhvbg; ?>;
	border:1px #<?php echo $elementhvbd; ?> solid;
}
/* End Custom Color Scheme */
</style><?php
	endif;

}