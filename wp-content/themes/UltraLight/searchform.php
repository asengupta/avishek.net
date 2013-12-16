			<h2>Search</h2>
			<form method="get" id="searchform" action="<?php bloginfo('url'); ?>">
				<div><input type="text" value="<?php echo wp_specialchars($s, 1); ?>" name="s" id="s" /></div>
			</form>